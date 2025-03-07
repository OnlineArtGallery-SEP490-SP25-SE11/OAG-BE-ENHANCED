import os
import logging
from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key")

# Swagger configuration
SWAGGER_URL = '/docs'  # Changed from /api/docs to avoid conflicts
API_URL = '/api/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Image Similarity Search API"
    }
)

# Register blueprints
logger.debug("Registering Swagger UI blueprint")
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Import routes after app initialization
logger.debug("Importing API blueprint")

from api import bp as api_bp
logger.debug("Registering API blueprint with prefix /api")
app.register_blueprint(api_bp, url_prefix='/api')

logger.debug("All blueprints registered")
logger.debug("Registered routes: %s", [str(rule) for rule in app.url_map.iter_rules()])