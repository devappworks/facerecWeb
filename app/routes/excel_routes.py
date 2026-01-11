from flask import Blueprint, jsonify, request
from app.controllers.excel_controller import ExcelController
from app.services.excel_service import ExcelService
from flask import current_app

excel_bp = Blueprint('excel', __name__, url_prefix='/api/excel')

@excel_bp.route('/process', methods=['GET'])
def process_excel():
    controller = ExcelController()
    result = controller.process_excel_and_fetch_images()
    return jsonify(result)

@excel_bp.route('/check-excel', methods=['GET'])
def check_excel_file():
    """
    API endpoint za proveru Excel fajla i pokretanje obrade u pozadini

    Proverava da li Excel fajl postoji i da li sadrži podatke.
    Ako je sve u redu, pokreće obradu u pozadini.

    Query parametri:
        country (str, obavezno): Zemlja za koju se traže poznate ličnosti
        occupation (str, opciono): Filter za određene zanimanja (npr. "Actor,Athlete")

    Returns:
        JSON: Status provere Excel fajla i pokretanja obrade
    """
    try:
        # Dobijanje parametra country iz URL-a
        country = request.args.get('country')
        occupation_filter = request.args.get('occupation')  # New parameter

        # Provera da li je country parametar prosleđen
        if not country:
            return jsonify({
                "success": False,
                "message": "Parametar 'country' je obavezan"
            }), 400

        # Inicijalizacija Excel servisa
        excel_service = ExcelService()

        # Poziv metode za proveru Excel fajla
        check_result = excel_service.check_excel_file()

        # Ako je provera uspešna, pokreni thread za obradu
        if check_result["success"]:
            result = excel_service.start_processing_thread(check_result, country, occupation_filter)
        else:
            result = check_result

        # Određivanje HTTP status koda na osnovu rezultata
        status_code = 200 if result["success"] else 400
        if "nije pronađen" in result.get("message", ""):
            status_code = 404

        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Greška prilikom provere Excel fajla: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Greška: {str(e)}"
        }), 500


@excel_bp.route('/occupations', methods=['GET'])
def get_occupations():
    """
    Get list of available occupations from occupation.xlsx

    Returns:
        JSON: List of available occupations
    """
    try:
        import os
        import pandas as pd

        occupation_file = os.getenv('EXCEL_FILE_PATH_OCCUPATION', 'storage/excel/occupation.xlsx')

        if not os.path.exists(occupation_file):
            return jsonify({
                "success": False,
                "message": "Occupation file not found"
            }), 404

        # Read Excel file
        df = pd.read_excel(occupation_file)

        if df.empty or 'Occupation' not in df.columns:
            return jsonify({
                "success": True,
                "data": {
                    "occupations": [],
                    "total": 0
                }
            }), 200

        # Get unique occupations
        occupations = df['Occupation'].dropna().unique().tolist()

        return jsonify({
            "success": True,
            "data": {
                "occupations": sorted(occupations),
                "total": len(occupations)
            }
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting occupations: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500 