import os
import sys
import os.path as osp
import tempfile
import shutil
from contextlib import asynccontextmanager
from typing import List
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# Add the pdf directory to the path
sys.path.append(osp.dirname(osp.abspath(__file__)))
print("sys.path appended")
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models

print("pdf_extract_kit.utils.config_loader loaded")
import pdf_extract_kit.tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the layout detection model on startup."""
    # Startup - skip model initialization on startup to avoid crashes
    # Models will be initialized lazily on first request
    print("FastAPI application starting...")
    print("Models will be initialized on first request.")
    
    yield
    
    # Shutdown (if needed)
    # Cleanup code can go here


app = FastAPI(lifespan=lifespan)

layout_detection_task = None
ocr_task = None
formula_recognition_task = None
formula_detection_task = None



def initialize_layout_detection(config_path: str = None):
    """Initialize the layout detection task."""
    global layout_detection_task
    
    if layout_detection_task is not None:
        return layout_detection_task
    
    # Get the directory where api.py is located
    api_dir = osp.dirname(osp.abspath(__file__))
    
    # Use default config path if not provided
    if config_path is None:
        config_path = osp.join(api_dir, 'configs', 'layout_detection.yaml')
    else:
        # Resolve relative paths relative to api.py's directory
        if not osp.isabs(config_path):
            config_path = osp.join(api_dir, config_path)
    
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Resolve relative model paths in config relative to api.py's directory
    if 'tasks' in config:
        for task_name, task_config in config['tasks'].items():
            if 'model_config' in task_config and 'model_path' in task_config['model_config']:
                model_path = task_config['model_config']['model_path']
                if not osp.isabs(model_path):
                    # Resolve relative to api.py's directory
                    task_config['model_config']['model_path'] = osp.join(api_dir, model_path)
    
    task_instances = initialize_tasks_and_models(config)
    
    if 'layout_detection' not in task_instances:
        raise ValueError("Layout detection task not found in configuration")
    
    layout_detection_task = task_instances['layout_detection']
    return layout_detection_task


def initialize_ocr(config_path: str = None):
    """Initialize the OCR task."""
    global ocr_task
    
    if ocr_task is not None:
        return ocr_task
    
    # Get the directory where api.py is located
    api_dir = osp.dirname(osp.abspath(__file__))
    
    # Use default config path if not provided
    if config_path is None:
        config_path = osp.join(api_dir, 'configs', 'ocr.yaml')
    else:
        # Resolve relative paths relative to api.py's directory
        if not osp.isabs(config_path):
            config_path = osp.join(api_dir, config_path)
    
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Resolve relative model paths in config relative to api.py's directory
    if 'tasks' in config:
        for task_name, task_config in config['tasks'].items():
            if 'model_config' in task_config:
                # Handle det_model_dir and rec_model_dir
                for path_key in ['det_model_dir', 'rec_model_dir']:
                    if path_key in task_config['model_config']:
                        model_path = task_config['model_config'][path_key]
                        if not osp.isabs(model_path):
                            task_config['model_config'][path_key] = osp.join(api_dir, model_path)
    
    task_instances = initialize_tasks_and_models(config)
    
    if 'ocr' not in task_instances:
        raise ValueError("OCR task not found in configuration")
    
    ocr_task = task_instances['ocr']
    return ocr_task


def initialize_formula_recognition(config_path: str = None):
    """Initialize the formula recognition task."""
    global formula_recognition_task
    
    if formula_recognition_task is not None:
        return formula_recognition_task
    
    # Get the directory where api.py is located
    api_dir = osp.dirname(osp.abspath(__file__))
    
    # Use default config path if not provided
    if config_path is None:
        config_path = osp.join(api_dir, 'configs', 'formula_recognition.yaml')
    else:
        # Resolve relative paths relative to api.py's directory
        if not osp.isabs(config_path):
            config_path = osp.join(api_dir, config_path)
    
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Resolve relative model paths in config relative to api.py's directory
    if 'tasks' in config:
        for task_name, task_config in config['tasks'].items():
            if 'model_config' in task_config:
                # Handle model_path and cfg_path
                for path_key in ['model_path', 'cfg_path']:
                    if path_key in task_config['model_config']:
                        model_path = task_config['model_config'][path_key]
                        if not osp.isabs(model_path):
                            # For cfg_path, it might be relative to pdf_extract_kit, so check if it exists
                            if path_key == 'cfg_path':
                                # Check if it's a relative path within the package
                                if not osp.exists(model_path):
                                    # Try resolving relative to api_dir
                                    resolved_path = osp.join(api_dir, model_path)
                                    if osp.exists(resolved_path):
                                        task_config['model_config'][path_key] = resolved_path
                                    # Otherwise keep original (might be package-relative)
                            else:
                                task_config['model_config'][path_key] = osp.join(api_dir, model_path)
    
    task_instances = initialize_tasks_and_models(config)
    
    if 'formula_recognition' not in task_instances:
        raise ValueError("Formula recognition task not found in configuration")
    
    formula_recognition_task = task_instances['formula_recognition']
    return formula_recognition_task


def initialize_formula_detection(config_path: str = None):
    """Initialize the formula detection task."""
    global formula_detection_task
    
    if formula_detection_task is not None:
        return formula_detection_task
    
    # Get the directory where api.py is located
    api_dir = osp.dirname(osp.abspath(__file__))
    
    # Use default config path if not provided
    if config_path is None:
        config_path = osp.join(api_dir, 'configs', 'formula_detection.yaml')
    else:
        # Resolve relative paths relative to api.py's directory
        if not osp.isabs(config_path):
            config_path = osp.join(api_dir, config_path)
    
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Resolve relative model paths in config relative to api.py's directory
    if 'tasks' in config:
        for task_name, task_config in config['tasks'].items():
            if 'model_config' in task_config and 'model_path' in task_config['model_config']:
                model_path = task_config['model_config']['model_path']
                if not osp.isabs(model_path):
                    # Resolve relative to api.py's directory
                    task_config['model_config']['model_path'] = osp.join(api_dir, model_path)
    
    task_instances = initialize_tasks_and_models(config)
    
    if 'formula_detection' not in task_instances:
        raise ValueError("Formula detection task not found in configuration")
    
    formula_detection_task = task_instances['formula_detection']
    return formula_detection_task


@app.get("/")
def read_root():
    return {"Hello": "World", "endpoints": ["/detect-layout", "/ocr", "/formula-recognition", "/formula-detection"]}

@app.post("/detect-layout")
async def detect_layout(files: List[UploadFile] = File(...)):
    """
    Upload one or more images and get layout detection results.
    
    Returns:
        list: List of detection results, each containing 'filename', 'boxes', 'classes', 'scores', and 'class_names'.
    """
    # Initialize model if not already done
    try:
        task = initialize_layout_detection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded")
    
    # Validate all files are images
    for file in files:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Get the model from the task
    model = task.model
    
    # Create temporary directory for results if visualization is enabled
    tmp_result_dir = None
    if model.visualize:
        tmp_result_dir = tempfile.mkdtemp()
        result_path = tmp_result_dir
    else:
        result_path = ""
    
    tmp_paths = []
    file_mapping = []  # Maps index in tmp_paths to original file index
    # Pre-allocate results list to maintain order
    all_results = [None] * len(files)
    
    try:
        # Process each file
        for file_idx, file in enumerate(files):
            try:
                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents)).convert('RGB')
                
                # Save to temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name, 'JPEG')
                    tmp_path = tmp_file.name
                    tmp_paths.append(tmp_path)
                    file_mapping.append(file_idx)
                
            except Exception as e:
                all_results[file_idx] = {
                    "filename": file.filename or f"image_{file_idx}.jpg",
                    "error": f"Error reading image: {str(e)}",
                    "boxes": [],
                    "classes": [],
                    "scores": [],
                    "class_names": []
                }
                continue
        
        if tmp_paths:
            # Predict using the model for all images at once
            image_ids = [os.path.splitext(os.path.basename(path))[0] for path in tmp_paths]
            results = model.predict(tmp_paths, result_path=result_path, image_ids=image_ids)
            
            # Process each result
            for idx, result in enumerate(results):
                original_file_idx = file_mapping[idx]
                filename = files[original_file_idx].filename or f"image_{original_file_idx}.jpg"
                
                try:
                    # Handle tensor or numpy array conversion
                    if hasattr(result.boxes.xyxy, 'cpu'):
                        boxes = result.boxes.xyxy.cpu().numpy()
                    elif hasattr(result.boxes.xyxy, 'numpy'):
                        boxes = result.boxes.xyxy.numpy()
                    else:
                        boxes = np.array(result.boxes.xyxy)
                    
                    if hasattr(result.boxes.cls, 'cpu'):
                        classes = result.boxes.cls.cpu().numpy()
                    elif hasattr(result.boxes.cls, 'numpy'):
                        classes = result.boxes.cls.numpy()
                    else:
                        classes = np.array(result.boxes.cls)
                    
                    if hasattr(result.boxes.conf, 'cpu'):
                        scores = result.boxes.conf.cpu().numpy()
                    elif hasattr(result.boxes.conf, 'numpy'):
                        scores = result.boxes.conf.numpy()
                    else:
                        scores = np.array(result.boxes.conf)
                    
                    # Convert to lists for JSON serialization
                    boxes_list = boxes.tolist()
                    classes_list = classes.tolist()
                    scores_list = scores.tolist()
                    
                    # Get class names
                    class_names = [model.id_to_names.get(int(cls), f"class_{int(cls)}") for cls in classes_list]
                    
                    all_results[original_file_idx] = {
                        "filename": filename,
                        "boxes": boxes_list,
                        "classes": classes_list,
                        "scores": scores_list,
                        "class_names": class_names
                    }
                
                except Exception as e:
                    all_results[original_file_idx] = {
                        "filename": filename,
                        "error": f"Error processing detection results: {str(e)}",
                        "boxes": [],
                        "classes": [],
                        "scores": [],
                        "class_names": []
                    }
        
        return JSONResponse(content=all_results)
    
    finally:
        # Clean up temporary files
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        # Clean up temporary result directory if it was created
        if tmp_result_dir and os.path.exists(tmp_result_dir):
            shutil.rmtree(tmp_result_dir)


@app.post("/formula-detection")
async def formula_detection(files: List[UploadFile] = File(...)):
    """
    Upload one or more images and get formula detection results.
    
    Returns:
        list: List of detection results, each containing 'filename', 'boxes', 'classes', 'scores', and 'class_names'.
    """
    # Initialize model if not already done
    try:
        task = initialize_formula_detection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize formula detection model: {str(e)}")
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded")
    
    # Validate all files are images
    for file in files:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Get the model from the task
    model = task.model
    
    # Create temporary directory for results if visualization is enabled
    tmp_result_dir = None
    if model.visualize:
        tmp_result_dir = tempfile.mkdtemp()
        result_path = tmp_result_dir
    else:
        result_path = ""
    
    tmp_paths = []
    file_mapping = []  # Maps index in tmp_paths to original file index
    # Pre-allocate results list to maintain order
    all_results = [None] * len(files)
    
    try:
        # Process each file
        for file_idx, file in enumerate(files):
            try:
                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents)).convert('RGB')
                
                # Save to temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name, 'JPEG')
                    tmp_path = tmp_file.name
                    tmp_paths.append(tmp_path)
                    file_mapping.append(file_idx)
                
            except Exception as e:
                all_results[file_idx] = {
                    "filename": file.filename or f"image_{file_idx}.jpg",
                    "error": f"Error reading image: {str(e)}",
                    "boxes": [],
                    "classes": [],
                    "scores": [],
                    "class_names": []
                }
                continue
        
        if tmp_paths:
            # Predict using the model for all images at once
            image_ids = [os.path.splitext(os.path.basename(path))[0] for path in tmp_paths]
            results = model.predict(tmp_paths, result_path=result_path, image_ids=image_ids)
            
            # Process each result
            for idx, result in enumerate(results):
                original_file_idx = file_mapping[idx]
                filename = files[original_file_idx].filename or f"image_{original_file_idx}.jpg"
                
                try:
                    # Handle tensor or numpy array conversion
                    if hasattr(result.boxes.xyxy, 'cpu'):
                        boxes = result.boxes.xyxy.cpu().numpy()
                    elif hasattr(result.boxes.xyxy, 'numpy'):
                        boxes = result.boxes.xyxy.numpy()
                    else:
                        boxes = np.array(result.boxes.xyxy)
                    
                    if hasattr(result.boxes.cls, 'cpu'):
                        classes = result.boxes.cls.cpu().numpy()
                    elif hasattr(result.boxes.cls, 'numpy'):
                        classes = result.boxes.cls.numpy()
                    else:
                        classes = np.array(result.boxes.cls)
                    
                    if hasattr(result.boxes.conf, 'cpu'):
                        scores = result.boxes.conf.cpu().numpy()
                    elif hasattr(result.boxes.conf, 'numpy'):
                        scores = result.boxes.conf.numpy()
                    else:
                        scores = np.array(result.boxes.conf)
                    
                    # Convert to lists for JSON serialization
                    boxes_list = boxes.tolist()
                    classes_list = classes.tolist()
                    scores_list = scores.tolist()
                    
                    # Get class names (inline or isolated)
                    class_names = [model.id_to_names.get(int(cls), f"class_{int(cls)}") for cls in classes_list]
                    
                    all_results[original_file_idx] = {
                        "filename": filename,
                        "boxes": boxes_list,
                        "classes": classes_list,
                        "scores": scores_list,
                        "class_names": class_names
                    }
                
                except Exception as e:
                    all_results[original_file_idx] = {
                        "filename": filename,
                        "error": f"Error processing detection results: {str(e)}",
                        "boxes": [],
                        "classes": [],
                        "scores": [],
                        "class_names": []
                    }
        
        return JSONResponse(content=all_results)
    
    finally:
        # Clean up temporary files
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        # Clean up temporary result directory if it was created
        if tmp_result_dir and os.path.exists(tmp_result_dir):
            shutil.rmtree(tmp_result_dir)


@app.post("/ocr")
async def ocr(files: List[UploadFile] = File(...)):
    """
    Upload one or more images and get OCR (text recognition) results.
    
    Returns:
        list: List of OCR results, each containing 'filename', 'texts' (list of detected text with category and text).
    """
    # Initialize OCR task if not already done
    try:
        task = initialize_ocr()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize OCR model: {str(e)}")
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded")
    
    # Validate all files are images
    for file in files:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Pre-allocate results list to maintain order
    all_results = [None] * len(files)
    
    try:
        # Process each file
        for file_idx, file in enumerate(files):
            try:
                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents)).convert('RGB')
                
                # Get filename for this image
                filename = file.filename or f"image_{file_idx}.jpg"
                
                # Run OCR prediction
                ocr_results = task.predict_image(image)
                
                # Format results: extract category and text for each detection
                texts = []
                for ocr_item in ocr_results:
                    texts.append({
                        "category": ocr_item.get("category_type", "text"),
                        "text": ocr_item.get("text", ""),
                        "score": ocr_item.get("score", 0.0),
                        "poly": ocr_item.get("poly", [])
                    })
                
                all_results[file_idx] = {
                    "filename": filename,
                    "texts": texts
                }
                
            except Exception as e:
                all_results[file_idx] = {
                    "filename": file.filename or f"image_{file_idx}.jpg",
                    "error": f"Error processing image: {str(e)}",
                    "texts": []
                }
        
        return JSONResponse(content=all_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/formula-recognition")
async def formula_recognition(files: List[UploadFile] = File(...)):
    """
    Upload one or more images and get formula recognition results (LaTeX strings).
    
    Returns:
        list: List of recognition results, each containing 'filename' and 'formula' (LaTeX string).
    """
    # Initialize formula recognition task if not already done
    try:
        task = initialize_formula_recognition()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize formula recognition model: {str(e)}")
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded")
    
    # Validate all files are images
    for file in files:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Pre-allocate results list to maintain order
    all_results = [None] * len(files)
    tmp_paths = []
    file_mapping = []  # Maps index in tmp_paths to original file index
    
    try:
        # Process each file - save to temporary files since model expects paths
        for file_idx, file in enumerate(files):
            try:
                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents)).convert('RGB')
                
                # Save to temporary file for processing (model expects file paths)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name, 'JPEG')
                    tmp_path = tmp_file.name
                    tmp_paths.append(tmp_path)
                    file_mapping.append(file_idx)
                
            except Exception as e:
                all_results[file_idx] = {
                    "filename": file.filename or f"image_{file_idx}.jpg",
                    "error": f"Error reading image: {str(e)}",
                    "formula": ""
                }
                continue
        
        if tmp_paths:
            # Get the model from the task
            model = task.model
            
            # Predict using the model (expects list of image paths)
            predictions = model.predict(tmp_paths, result_path="")
            
            # Map predictions back to files
            for pred_idx, prediction in enumerate(predictions):
                if pred_idx < len(file_mapping):
                    original_file_idx = file_mapping[pred_idx]
                    filename = files[original_file_idx].filename or f"image_{original_file_idx}.jpg"
                    all_results[original_file_idx] = {
                        "filename": filename,
                        "formula": prediction
                    }
            
            # Handle cases where some predictions might be missing (model errors)
            for idx, file_idx in enumerate(file_mapping):
                if all_results[file_idx] is None:
                    filename = files[file_idx].filename or f"image_{file_idx}.jpg"
                    all_results[file_idx] = {
                        "filename": filename,
                        "error": "No prediction returned from model",
                        "formula": ""
                    }
        
        return JSONResponse(content=all_results)
    
    finally:
        # Clean up temporary files
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)