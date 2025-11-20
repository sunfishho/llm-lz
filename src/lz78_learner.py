"""Core neural architecture for LLM-guided LZ78 learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from lz78 import Sequence as LZ78Sequence

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from config import ModelConfig
from data_utils import LogitsCache, bytes_to_lz78_sequence

logger = logging.getLogger(__name__)


class LLMLogitsExtractor:
    """Extract logits from LLM for given text sequences."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 4,
        cache: Optional[LogitsCache] = None
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache = cache
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use AutoModelForCausalLM to get proper logits
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_hidden_states=False,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        except Exception as e:
            logger.warning(f"Failed to load as CausalLM, falling back to AutoModel: {e}")
            self.model = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=False,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded LLM {model_name} on device {self.device}")
    
    def extract_logits(self, texts: List[str]) -> torch.Tensor:
        """Extract logits for a batch of texts."""
        # Check cache first
        if self.cache is not None:
            cached_logits = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    cached_logits.append(cached)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if len(uncached_texts) == 0:
                # All cached
                return torch.stack(cached_logits)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_logits = []
        
        # Process uncached texts
        if len(uncached_texts) > 0:
            new_logits = self._extract_logits_batch(uncached_texts)
            
            # Cache new logits
            if self.cache is not None:
                for text, logits in zip(uncached_texts, new_logits):
                    self.cache.set(text, logits)
        
        # Combine cached and new logits
        if self.cache is not None and len(cached_logits) > 0:
            all_logits = [None] * len(texts)
            for i, logits in zip(uncached_indices, new_logits):
                all_logits[i] = logits
            for i, logits in enumerate(cached_logits):
                if all_logits[i] is None:
                    all_logits[i] = logits
            return torch.stack(all_logits)
        else:
            return new_logits
    
    def _extract_logits_batch(self, texts: List[str]) -> torch.Tensor:
        """Extract logits for a batch of texts (internal method)."""
        all_logits = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get logits from the model
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(outputs.last_hidden_state)
                else:
                    # Fallback: use last hidden state (this will cause dimension mismatch)
                    logits = outputs.last_hidden_state
                
                # Remove padding tokens
                attention_mask = inputs['attention_mask']
                logits = logits * attention_mask.unsqueeze(-1).float()
                
                all_logits.append(logits.cpu())
        
        return torch.cat(all_logits, dim=0)


class LZ78PretrainGenerator(nn.Module):
    """Neural network that generates synthetic pretraining sequences from LLM logits and training data."""
    
    def __init__(
        self,
        llm_vocab_size: int,
        alphabet_size: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        pretrain_sequence_length: int = 1000,
        gumbel_temperature: float = 1.0,
        gumbel_hard: bool = True,
        max_training_data_length: int = 1024
    ):
        super().__init__()
        
        self.llm_vocab_size = llm_vocab_size
        self.alphabet_size = alphabet_size
        self.pretrain_sequence_length = pretrain_sequence_length
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_hard = gumbel_hard
        self.max_training_data_length = max_training_data_length
        
        # Project LLM logits to hidden dimension
        self.llm_projection = nn.Linear(llm_vocab_size, hidden_dim)
        
        # Embedding layer for training data (maps alphabet indices to hidden_dim)
        self.training_data_embedding = nn.Embedding(alphabet_size, hidden_dim)
        
        # LLM sequence encoder (processes the sequence of projected logits)
        self.llm_sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Training data sequence encoder (processes training data sequences)
        self.training_data_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Cross-attention mechanism to combine LLM and training data features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Global pooling to get sequence-level representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Generator network that outputs sequence parameters
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pretrain_sequence_length * alphabet_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        llm_logits: torch.Tensor,
        training_data: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic pretraining sequence from LLM logits and optional training data.
        
        Args:
            llm_logits: [batch_size, seq_len, llm_vocab_size]
            training_data: [batch_size, training_seq_len] (optional, discrete indices)
            temperature: Temperature for Gumbel-Softmax sampling
            
        Returns:
            synthetic_sequence: [batch_size, pretrain_sequence_length] (discrete)
            log_probs: [batch_size, pretrain_sequence_length] (for training)
        """
        batch_size, seq_len, _ = llm_logits.shape
        
        # Project LLM logits to hidden dimension
        llm_projected = self.llm_projection(llm_logits)  # [batch_size, seq_len, hidden_dim]
        
        # Encode LLM sequence
        llm_encoded = self.llm_sequence_encoder(llm_projected)  # [batch_size, seq_len, hidden_dim]
        
        if training_data is not None:
            # Process training data
            training_data = training_data.long()  # Ensure integer type
            # Truncate if too long
            if training_data.size(1) > self.max_training_data_length:
                training_data = training_data[:, :self.max_training_data_length]
            
            # Embed training data
            training_embedded = self.training_data_embedding(training_data)  # [batch_size, training_seq_len, hidden_dim]
            
            # Encode training data sequence
            training_encoded = self.training_data_encoder(training_embedded)  # [batch_size, training_seq_len, hidden_dim]
            
            # Cross-attention: LLM queries, training data keys/values
            attended_features, _ = self.cross_attention(
                query=llm_encoded,
                key=training_encoded,
                value=training_encoded
            )  # [batch_size, seq_len, hidden_dim]
            
            # Combine LLM and training data features
            combined_features = llm_encoded + attended_features  # Residual connection
        else:
            # Use only LLM features if no training data provided
            combined_features = llm_encoded
        
        # Global pooling
        pooled = self.global_pool(combined_features.transpose(1, 2))  # [batch_size, hidden_dim, 1]
        pooled = pooled.squeeze(-1)  # [batch_size, hidden_dim]
        
        # Generate sequence parameters
        sequence_params = self.generator(pooled)  # [batch_size, pretrain_sequence_length * alphabet_size]
        sequence_params = sequence_params.view(
            batch_size, self.pretrain_sequence_length, self.alphabet_size
        )
        
        # Convert to probabilities
        logits = sequence_params  # [batch_size, pretrain_sequence_length, alphabet_size]
        
        # Sample using Gumbel-Softmax
        temp = temperature if temperature is not None else self.gumbel_temperature
        synthetic_sequence, log_probs = self._gumbel_softmax_sample(logits, temp)
        
        return synthetic_sequence, log_probs
    
    def _gumbel_softmax_sample(
        self,
        logits: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from Gumbel-Softmax distribution."""
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        gumbel_logits = (logits + gumbel_noise) / temperature
        
        # Softmax
        probs = F.softmax(gumbel_logits, dim=-1)
        
        if self.gumbel_hard and self.training:
            # Straight-through estimator
            _, hard_indices = torch.max(probs, dim=-1)
            hard_probs = torch.zeros_like(probs)
            hard_probs.scatter_(-1, hard_indices.unsqueeze(-1), 1.0)
            
            # Use hard samples for forward pass, soft for backward pass
            synthetic_sequence = hard_probs + probs - probs.detach()
            log_probs = torch.log(probs + 1e-8)
        else:
            synthetic_sequence = probs
            log_probs = torch.log(probs + 1e-8)
        
        return synthetic_sequence, log_probs
    
    def generate_discrete_sequence(
        self,
        llm_logits: torch.Tensor,
        training_data: Optional[torch.Tensor] = None,
        temperature: float = 0.1
    ) -> List[int]:
        """Generate a discrete integer sequence for LZ78 training."""
        with torch.no_grad():
            synthetic_sequence, _ = self.forward(llm_logits, training_data, temperature)
            
            # Convert to discrete indices
            if synthetic_sequence.dim() == 3:  # [batch_size, seq_len, alphabet_size]
                discrete_indices = torch.argmax(synthetic_sequence, dim=-1)
                indices = discrete_indices[0].cpu().tolist()  # Return first batch
            else:
                discrete_indices = torch.argmax(synthetic_sequence, dim=-1)
                indices = discrete_indices.cpu().tolist()
            
            # Return indices directly (they should be in range [0, alphabet_size-1])
            # The LZ78 module expects indices in this range, not ASCII values
            return indices


class CompressionLoss(nn.Module):
    """Differentiable approximation of compression loss using LZ78 test loss."""
    
    def __init__(self, alphabet_size: int = 256, gamma: float = 0.5):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.gamma = gamma
        
    def forward(
        self,
        synthetic_sequence: torch.Tensor,
        validation_sequence: List[int],
        lz78_spa
    ) -> torch.Tensor:
        """
        Compute compression loss.
        
        Args:
            synthetic_sequence: [batch_size, pretrain_sequence_length, alphabet_size] (soft)
            validation_sequence: List of integers for validation
            lz78_spa: LZ78SPA instance for computing test loss
            
        Returns:
            loss: Scalar tensor
        """
        # Convert soft sequence to discrete for LZ78 training
        discrete_sequence = torch.argmax(synthetic_sequence, dim=-1)[0].cpu().tolist()
        
        # Train LZ78 SPA on synthetic sequence
        synthetic_lz78_seq = LZ78Sequence(discrete_sequence, alphabet_size=self.alphabet_size)
        
        # Reset and train on synthetic sequence
        lz78_spa.reset_state()
        train_result = lz78_spa.train_on_block(synthetic_lz78_seq)
        
        # Compute test loss on validation sequence
        val_lz78_seq = LZ78Sequence(validation_sequence, alphabet_size=self.alphabet_size)
        test_result = lz78_spa.compute_test_loss(val_lz78_seq)
        
        # Use average log loss as surrogate for compression
        avg_log_loss = test_result['average_log_loss']
        
        return torch.tensor(avg_log_loss, dtype=torch.float32, requires_grad=True)


class LZ78Learner:
    """Main class that coordinates the learning process."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Initialize components
        self.llm_extractor = LLMLogitsExtractor(
            model_name=config.llm_model_name,
            device=config.llm_device,
            max_length=config.llm_max_length,
            batch_size=config.llm_batch_size
        )
        
        self.pretrain_generator = LZ78PretrainGenerator(
            llm_vocab_size=self.llm_extractor.tokenizer.vocab_size,
            alphabet_size=config.alphabet_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pretrain_sequence_length=config.pretrain_sequence_length,
            gumbel_temperature=config.gumbel_temperature,
            gumbel_hard=config.gumbel_hard
        )
        
        self.compression_loss = CompressionLoss(
            alphabet_size=config.alphabet_size,
            gamma=0.5  # Will be set by LZ78SPA
        )
        
        logger.info("LZ78Learner initialized successfully")
    
    def extract_llm_features(self, texts: List[str]) -> torch.Tensor:
        """Extract LLM logits for texts."""
        return self.llm_extractor.extract_logits(texts)
    
    def prepare_training_data(self, text_bytes_list: List[bytes]) -> torch.Tensor:
        """Convert training data bytes to tensor format for the generator."""
        from data_utils import bytes_to_lz78_sequence
        
        # Handle empty input
        if not text_bytes_list:
            return torch.empty(0, 0, dtype=torch.long)
        
        # Convert bytes to LZ78 sequences
        sequences = []
        for text_bytes in text_bytes_list:
            sequence = bytes_to_lz78_sequence(text_bytes, self.config.alphabet_size)
            # Truncate if too long
            if len(sequence) > self.config.max_training_data_length:
                sequence = sequence[:self.config.max_training_data_length]
            sequences.append(sequence)
        
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                # Pad with zeros (or could use a special padding token)
                padded_seq = seq + [0] * (max_len - len(seq))
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences, dtype=torch.long)
    
    def generate_pretrain_sequence(
        self,
        llm_logits: torch.Tensor,
        training_data: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic pretraining sequence."""
        return self.pretrain_generator(llm_logits, training_data, temperature)
    
    def compute_compression_loss(
        self,
        synthetic_sequence: torch.Tensor,
        validation_sequence: List[int],
        lz78_spa
    ) -> torch.Tensor:
        """Compute compression loss."""
        return self.compression_loss(synthetic_sequence, validation_sequence, lz78_spa)
