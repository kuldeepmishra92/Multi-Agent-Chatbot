import os
import sys
import uuid
import time
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from flask_cors import CORS
from flask_compress import Compress

# Ensure the parent directory is in the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.validate_config()

from rag.document_loader import load_and_chunk_pdf
from rag.vector_store    import add_documents, get_document_count, clear_store, get_unique_sources, delete_source
from rag.retriever       import invalidate_bm25_cache
from orchestrator.graph  import run_chat
from utils.logger        import get_logger

logger = get_logger(__name__)

app = Flask(__name__, static_folder='../static', template_folder='../templates')
CORS(app)
Compress(app)

# In-memory session store (for demo purposes; real app would use a DB)
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('../static', path)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        chunks = get_document_count()
    except:
        chunks = 0
    return jsonify({
        "total_chunks": chunks
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    indexed_count = 0
    total_chunks = 0
    
    for f in files:
        if f.filename.endswith('.pdf'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    f.save(tmp.name)
                    tmp_path = tmp.name
                
                chunks = load_and_chunk_pdf(tmp_path)
                for ch in chunks:
                    ch.metadata["source"] = f.filename
                
                count = add_documents(chunks)
                invalidate_bm25_cache()
                os.unlink(tmp_path)
                
                indexed_count += 1
                total_chunks += count
                logger.info(f"Indexed: {f.filename} ({count} chunks)")
            except Exception as e:
                logger.error(f"Error indexing {f.filename}: {e}")
                return jsonify({"error": str(e)}), 500
                
    return jsonify({
        "message": f"Successfully indexed {indexed_count} file(s)",
        "total_chunks": total_chunks
    })

@app.route('/api/clear', methods=['POST'])
def clear_database():
    try:
        clear_store()
        invalidate_bm25_cache()
        logger.info("Knowledge base cleared via API.")
        return jsonify({"message": "Knowledge base cleared successfully", "total_chunks": 0})
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    try:
        sources = get_unique_sources()
        return jsonify({"documents": sources})
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete_document', methods=['DELETE'])
def delete_document():
    data = request.json
    source_name = data.get('filename')
    if not source_name:
        return jsonify({"error": "Filename is required"}), 400
    
    try:
        success = delete_source(source_name)
        if success:
            invalidate_bm25_cache()
            return jsonify({"message": f"Document '{source_name}' deleted successfully"})
        else:
            return jsonify({"error": "Failed to delete document"}), 500
    except Exception as e:
        logger.error(f"Error during document deletion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message')
    session_id = data.get('session_id', str(uuid.uuid4())[:8])
    
    if not query:
        return jsonify({"error": "Message is required"}), 400

    def generate():
        start = time.perf_counter()
        # In a real streaming implementation with LangGraph, we'd wrap the graph 
        # but for simplicity we'll follow our existing logic which split by words
        result = run_chat(query=query, session_id=session_id)
        elapsed = round(time.perf_counter() - start, 2)
        
        # Metadata header (to inform the frontend about agent used and latency)
        # We'll use a specific prefix to distinguish metadata from content
        yield f"METADATA:{{\"agent\":\"{result['agent_used']}\", \"latency\":{elapsed}}}\n"
        
        words = result["response"].split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
            time.sleep(0.015) # Simulated streaming for premium feel

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
