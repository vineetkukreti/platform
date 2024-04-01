import json
import argparse
import logging
import traceback
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
import model as embeddings

def to_array(emb):
    """Converts the embedding tensor to a list."""
    return [v.item() for v in emb]

class EmbeddingsServer(BaseHTTPRequestHandler):
    """Handles HTTP requests for embedding-related operations."""
    
    def __init__(self, emb_service, *args, **kwargs):
        """Initialize the EmbeddingsServer with an EmbeddingService instance."""
        self.emb_service = emb_service
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """Handles POST requests."""
        try:
            if self.path == '/embeddings':
                self.send_embeddings()
                return
            elif self.path == '/completion':
                self.send_completion()
                return
            elif self.path == '/compare':
                self.send_compare()
                return
        except Exception as e:
            logging.error('Failed to process request: %s', e)
            self.send_error(500, str(e))

    def send_embeddings(self):
        """Handles '/embeddings' endpoint."""
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        js_body = json.loads(data)
        model = js_body["model"]
        try:
            embeddings = self.emb_service.embeddings(js_body["input"])
            emb = to_array(embeddings[0])
            obj = {
                "data": [{
                    "embedding": emb,
                    "size": len(emb)
                }],
                "model": model,
                "usage": {
                    "prompt_tokens": embeddings[1],
                    "total_tokens": 1
                }
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(obj), "utf-8"))
        except Exception as e:
            logging.error('Error processing embeddings: %s', e)
            self.send_error(400, str(e))
            traceback.print_exc()

    def send_completion(self):
        """Handles '/completion' endpoint."""
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        js_body = json.loads(data)
        completion = self.emb_service.completion(js_body["input"], max_length=js_body["max_length"],
                                                 temperature=js_body["temperature"])
        model = js_body["model"]
        obj = {
            "data": [{
                "completion": completion
            }],
            "model": model
        }
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(json.dumps(obj), "utf-8"))

    def send_compare(self):
        """Handles '/compare' endpoint."""
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        js_body = json.loads(data)
        emb1 = self.emb_service.embeddings(js_body["input"])
        emb2 = self.emb_service.embeddings(js_body["compare"])
        model = js_body["model"]
        e1 = to_array(emb1[0])
        e2 = to_array(emb2[0])
        similarity = self.emb_service.compare(emb1[0], emb2[0]).item()
        obj = {
            "similarity": similarity,
            "input": e1,
            "input_len": len(e1),
            "compare": e2,
            "compare_len": len(e2),
            "model": model
        }
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(json.dumps(obj), "utf-8"))

def main():
    """Main function to start the embedding service."""
    parser = argparse.ArgumentParser(prog='Embedding\'s service')
    parser.add_argument('--model', default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument('--host', default="0.0.0.0")
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--port', type=int, default=4070)
    args = parser.parse_args()

    host_name = args.host
    server_port = args.port
    device = args.device
    model = args.model

    logging.basicConfig(level=logging.INFO)

    logging.info('Loading model: %s on device: %s', model, device)
    emb_service = embeddings.EmbeddingService(model, device)

    web_server = HTTPServer((host_name, server_port), partial(EmbeddingsServer, emb_service),
                            bind_and_activate=False)
    web_server.allow_reuse_address = True
    web_server.daemon_threads = True

    web_server.server_bind()
    web_server.server_activate()
    logging.info("Embedding service started at http://%s:%s", host_name, server_port)

    try:
        web_server.serve_forever()
    except KeyboardInterrupt:
        pass

    web_server.server_close()
    logging.info("Server stopped.")

if __name__ == "__main__":
    main()
