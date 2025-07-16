from flask import Blueprint, request, jsonify
from controller import make_query

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/api/query", methods=["POST"])
def query():
    try:
        body = request.get_json()

        sql_query = body.get("query")
        limit = int(body.get("limit", 100))
        offset = int(body.get("offset", 0))

        if not sql_query:
            return jsonify({"status": "error", "message": "Falta el campo 'query'"}), 400

        status, data, next_offset = make_query(sql_query, limit, offset)

        return jsonify({
            "status": status,
            "data": data,
            "next_offset": next_offset
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
