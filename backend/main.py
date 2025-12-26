"""
Backend API for PII Detection Demo
FastAPI server with CORS support for frontend communication
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Optional
import uvicorn
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="PII Detection API",
    description="API for detecting Personally Identifiable Information (PII) in Vietnamese text using mBERT",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
id2label = None
label2id = None

# Model configuration
# Paths relative to project root (not backend directory)
MODEL_PATH = r"D:\CODE\PII\pii_multi"  # Main model path
FALLBACK_MODEL_PATH = "../model/pii_mBert_case"  # Fallback if main model not found


class TextInput(BaseModel):
    """Input model for text prediction"""
    text: str


class TokenPrediction(BaseModel):
    """Model for token prediction result"""
    token: str
    label: str
    label_id: int


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    text: str
    tokens: List[TokenPrediction]
    has_pii: bool
    entities: Dict[str, List[str]]  # Grouped entities by type
    stats: Dict[str, int]  # Statistics about detected entities


def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer, device, id2label, label2id
    
    # Resolve paths - try both relative to backend/ and project root
    possible_paths = [
        MODEL_PATH,
        FALLBACK_MODEL_PATH,
        MODEL_PATH.replace("../", ""),  # If running from project root
        FALLBACK_MODEL_PATH.replace("../", ""),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        raise FileNotFoundError(
            f"Model not found. Tried: {', '.join(possible_paths[:2])}. "
            "Please ensure the model is trained and saved at 'model/pii_vn' or 'model/pii_mBert_case'."
        )
    
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from: {model_path}")
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Get label mapping from model config
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    print(f"✅ Model loaded successfully on device: {device}")
    print(f"📊 Total labels: {len(id2label)}")


def preprocess_text(text: str, start_idx: int = 0, end_idx: int = None) -> Dict[str, torch.Tensor]:
    """
    Preprocess input text following the same logic as training preprocessing.
    Returns tensors ready for the model plus the token sequence (without special tokens).
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded")

    # Tokenize text
    tokens = tokenizer.tokenize(text)

    # Convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Apply slicing if specified (for chunking)
    if end_idx is not None:
        tokens = tokens[start_idx:end_idx]
        input_ids = input_ids[start_idx:end_idx]

    # Cap length to leave room for CLS/SEP
    max_len = 512 - 2
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        tokens = tokens[:max_len]

    # Add special tokens
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    # Convert to tensors
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)

    return {
        "tokens": tokens,  # without CLS/SEP
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
    }


def split_text_into_chunks(text: str, max_tokens: int = 510, overlap: int = 50) -> List[tuple]:
    """
    Split long text into overlapping chunks for processing.
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded")
    
    # Tokenize the entire text
    all_tokens = tokenizer.tokenize(text)
    total_tokens = len(all_tokens)
    
    # If text fits in one chunk, return single chunk
    if total_tokens <= max_tokens:
        return [(0, total_tokens)]
    
    chunks = []
    start_idx = 0
    
    while start_idx < total_tokens:
        end_idx = min(start_idx + max_tokens, total_tokens)
        chunks.append((start_idx, end_idx))
        
        # Move start_idx forward, accounting for overlap
        if end_idx >= total_tokens:
            break
        start_idx = end_idx - overlap
    
    return chunks


def merge_chunk_predictions(chunk_results: List[Dict], chunks: List[tuple], overlap: int = 50) -> Dict:
    """
    Merge predictions from multiple chunks, handling overlaps.
    """
    if not chunk_results:
        return {"tokens": [], "pred_ids": []}
    
    # If only one chunk, return it directly
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    merged_tokens = []
    merged_pred_ids = []
    
    overlap_half = overlap // 2
    
    for i, (chunk_result, (start_idx, end_idx)) in enumerate(zip(chunk_results, chunks)):
        tokens = chunk_result["tokens"]
        pred_ids = chunk_result["pred_ids"]
        
        if i == 0:
            # First chunk: take all tokens except the last overlap_half
            if len(chunks) > 1:
                take_until = len(tokens) - overlap_half
            else:
                take_until = len(tokens)
            
            merged_tokens.extend(tokens[:take_until])
            merged_pred_ids.extend(pred_ids[:take_until])
        
        elif i == len(chunks) - 1:
            # Last chunk: skip the overlap region at the start
            merged_tokens.extend(tokens[overlap_half:])
            merged_pred_ids.extend(pred_ids[overlap_half:])
        
        else:
            # Middle chunks: skip overlap at start, take all except last overlap_half
            take_until = len(tokens) - overlap_half
            merged_tokens.extend(tokens[overlap_half:take_until])
            merged_pred_ids.extend(pred_ids[overlap_half:take_until])
    
    return {
        "tokens": merged_tokens,
        "pred_ids": merged_pred_ids
    }


def group_entities(tokens: List[str], labels: List[str]) -> Dict[str, List[str]]:
    """
    Group tokens by entity type to form complete entities.
    """
    entities = {}
    current_entity = None
    current_tokens = []
    
    for token, label in zip(tokens, labels):
        if label == "O":
            # Save current entity if exists
            if current_entity and current_tokens:
                entity_text = "".join(current_tokens).replace("##", "")
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(entity_text)
            current_entity = None
            current_tokens = []
        elif label.startswith("B-"):
            # Save previous entity if exists
            if current_entity and current_tokens:
                entity_text = "".join(current_tokens).replace("##", "")
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(entity_text)
            # Start new entity
            current_entity = label[2:]  # Remove "B-" prefix
            current_tokens = [token]
        elif label.startswith("I-"):
            entity_type = label[2:]  # Remove "I-" prefix
            if entity_type == current_entity:
                current_tokens.append(token)
            else:
                # Entity type changed, save previous and start new
                if current_entity and current_tokens:
                    entity_text = "".join(current_tokens).replace("##", "")
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append(entity_text)
                current_entity = entity_type
                current_tokens = [token]
    
    # Save last entity if exists
    if current_entity and current_tokens:
        entity_text = "".join(current_tokens).replace("##", "")
        if current_entity not in entities:
            entities[current_entity] = []
        entities[current_entity].append(entity_text)
    
    return entities


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  API will start but /predict endpoint will not work until model is loaded")
        # Don't raise - allow API to start for health checks


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PII Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "device": str(device) if device else "not loaded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device) if device else None,
        "total_labels": len(id2label) if id2label else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_pii(input_data: TextInput):
    """
    Predict PII labels for each token in the input text.
    Automatically handles texts longer than 512 tokens using sliding window chunking.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and ensure model files exist."
        )
    
    if not input_data.text or not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Check if text needs chunking
        all_tokens = tokenizer.tokenize(input_data.text)
        max_tokens_per_chunk = 510  # Leave room for CLS/SEP
        overlap = 50  # Overlap between chunks
        
        if len(all_tokens) <= max_tokens_per_chunk:
            # Process normally if text fits in one chunk
            processed = preprocess_text(input_data.text)
            tokens = processed["tokens"]

            model_inputs = {
                "input_ids": processed["input_ids"],
                "attention_mask": processed["attention_mask"],
            }

            if "token_type_ids" in tokenizer.model_input_names:
                model_inputs["token_type_ids"] = torch.zeros_like(processed["input_ids"])

            # Run inference
            with torch.no_grad():
                outputs = model(**model_inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            
            pred_ids = predictions[0].cpu().numpy()[1: len(tokens) + 1]
            
        else:
            # Process in chunks for long texts
            chunks = split_text_into_chunks(input_data.text, max_tokens_per_chunk, overlap)
            chunk_results = []
            
            for start_idx, end_idx in chunks:
                # Preprocess this chunk
                processed = preprocess_text(input_data.text, start_idx, end_idx)
                chunk_tokens = processed["tokens"]

                model_inputs = {
                    "input_ids": processed["input_ids"],
                    "attention_mask": processed["attention_mask"],
                }

                if "token_type_ids" in tokenizer.model_input_names:
                    model_inputs["token_type_ids"] = torch.zeros_like(processed["input_ids"])

                # Run inference for this chunk
                with torch.no_grad():
                    outputs = model(**model_inputs)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                
                chunk_pred_ids = predictions[0].cpu().numpy()[1: len(chunk_tokens) + 1]
                
                chunk_results.append({
                    "tokens": chunk_tokens,
                    "pred_ids": chunk_pred_ids
                })
            
            # Merge chunk results
            merged = merge_chunk_predictions(chunk_results, chunks, overlap)
            tokens = merged["tokens"]
            pred_ids = merged["pred_ids"]
        
        # Map predictions to labels
        token_predictions = []
        labels = []
        has_pii = False
        
        for token, pred_id in zip(tokens, pred_ids):
            # Get the predicted label
            label = id2label.get(int(pred_id), "O")
            labels.append(label)
            
            # Check if this is a PII label (not "O")
            if label != "O":
                has_pii = True
            
            token_predictions.append({
                "token": token,
                "label": label,
                "label_id": int(pred_id)
            })
        
        # Group entities
        entities = group_entities(tokens, labels)
        
        # Calculate statistics
        stats = {entity_type: len(values) for entity_type, values in entities.items()}
        stats["total_entities"] = sum(stats.values())
        stats["total_tokens"] = len(tokens)
        stats["pii_tokens"] = sum(1 for label in labels if label != "O")
        
        return PredictionResponse(
            text=input_data.text,
            tokens=token_predictions,
            has_pii=has_pii,
            entities=entities,
            stats=stats
        )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    # Run from backend directory, so model path should be relative to project root
    # Adjust if running from different location
    uvicorn.run(app, host="0.0.0.0", port=8000)

