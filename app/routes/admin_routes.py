from flask import Blueprint, jsonify
from app.services.name_mapping_service import NameMappingService

admin_routes = Blueprint('admin_routes', __name__)

@admin_routes.route('/name-mappings', methods=['GET'])
def get_name_mappings():
    """
    Vraća sva mapiranja između normalizovanih i originalnih imena.
    """
    mappings = NameMappingService.get_all_mappings()
    return jsonify({
        "status": "success",
        "count": len(mappings),
        "mappings": mappings
    }) 