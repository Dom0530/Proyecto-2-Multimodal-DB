from pyparsing import (
    Word,
    alphas,
    alphanums,
    Keyword,
    CaselessKeyword,
    Suppress,
    Group,
    delimitedList,
    Optional,
    OneOrMore,
    nums,
    quotedString,
    removeQuotes,
    ParserElement,
    Combine,
)

# Para acelerar el parsing
ParserElement.enablePackrat()

# 1. Definir literales básicos y tokens
# -------------------------------------

# Identificador: empieza con letra, puede tener letras, dígitos o guiones bajos
identifier = Word(alphas, alphanums + "_").setName("identifier")

# Número natural (sin signo)
nat_number = Word(nums).setName("nat_number")

# Número (por simplicidad, mismo que nat_number; podría expandirse a decimales etc.)
number = Word(nums).setName("number")

# Cadena entre comillas simples o dobles
string_literal = quotedString.setParseAction(removeQuotes).setName("string")

# Carácter (entre comillas simples) – se acepta un solo carácter
char_literal = (
    Combine("'"
            + Word(alphanums + "!@#$%^&*()_+-=[]{}|;:,.<>/?", exact=1)
            + "'")
    .setParseAction(lambda t: t[0][1])  # extrae el carácter sin comillas
    .setName("char")
)

# Operadores de condición: =, @@, <->
eq_op = Keyword("=").setName("=")
match_op = Keyword("@@").setName("@@")
dist_op = Keyword("<->").setName("<->")
cond_op = (eq_op | match_op | dist_op).setName("cond_op")

# Tipos de datos (para CREATE TABLE)
INT     = CaselessKeyword("INT")
FLOAT   = CaselessKeyword("FLOAT")
CHAR    = CaselessKeyword("CHAR")
VARCHAR = CaselessKeyword("VARCHAR")

# Definición de tipo CHAR(n) o VARCHAR(n)
char_type    = (CHAR + Suppress("(") + nat_number("length") + Suppress(")")).setParseAction(lambda t: f"CHAR({t.length})")
varchar_type = (VARCHAR + Suppress("(") + nat_number("length") + Suppress(")")).setParseAction(lambda t: f"VARCHAR({t.length})")

# Unión de tipos válidos (puede agregarse más tipos si es necesario)
row_type = (INT | FLOAT | char_type | varchar_type).setName("row_type")

# Índices permitidos: SPIMI o IVEC
SPIMI = CaselessKeyword("SPIMI")
IVEC  = CaselessKeyword("IVEC")
index_type = (SPIMI | IVEC).setName("index_type")


# 2. Construir reglas más complejas
# ---------------------------------

# 2.1 Column definition: <row_name> <row_type>
column_def = Group(identifier("col_name") + row_type("col_type"))\
    .setName("column_def")

# Lista de definiciones de columnas: <column_def> [, <column_def>]*
column_defs = delimitedList(column_def).setName("column_defs")

# 2.2 Valores para INSERT: <value> [, <value>]*
# Aquí <value> puede ser número o cadena
value = (number | string_literal).setName("value")
values_list = Group(delimitedList(value))("values")

# 2.3 Condición: <row_name> <cond_op> <value>
condition = Group(
    identifier("left")
    + cond_op("op")
    + value("right")
).setName("condition")

# 2.4 <nat_number> para LIMIT
limit_clause = CaselessKeyword("LIMIT") + nat_number("limit")

# 2.5 <table_name>, <index_name>, <row_name> usan 'identifier'
table_name = identifier("table")
index_name = identifier("index")
row_name   = identifier("column")


# 3. Definición de cada sentencia
# -------------------------------

# 3.1 SELECT: SELECT * FROM <table_name> [WHERE <condition>] [LIMIT <nat_number>];
SELECT = CaselessKeyword("SELECT")
FROM   = CaselessKeyword("FROM")
WHERE  = CaselessKeyword("WHERE")

# Lista de columnas: columna1, columna2, ...
column_list = Group(delimitedList(identifier))("columns")

from pyparsing import Literal

select_stmt = (
    SELECT.suppress()
    + (Literal("*").setResultsName("all_columns") | column_list)
    + FROM.suppress()
    + table_name
    + Optional(WHERE.suppress() + condition)("where_cond")
    + Optional(limit_clause)
).setParseAction(lambda t: {
    "tipo": "SELECT",
    "tabla": t.table,
    "columnas": ["*"] if "all_columns" in t else t.columns.asList(),
    "where": t.where_cond.asList() if t.get("where_cond") else None,
    "limit": int(t.limit) if t.get("limit") else None
})


# 3.2 CREATE TABLE
CREATE = CaselessKeyword("CREATE")
TABLE  = CaselessKeyword("TABLE")
FROM   = CaselessKeyword("FROM")
FILE   = CaselessKeyword("FILE")
DELIMITED = CaselessKeyword("DELIMITED")

# Opción A: CREATE TABLE <table_name> FROM FILE <string> DELIMITED <char>
create_table_from_file = (
    CREATE.suppress()
    + TABLE.suppress()
    + table_name
    + FROM.suppress()
    + FILE.suppress()
    + string_literal("filename")
    + DELIMITED.suppress()
    + char_literal("delimiter")
).setParseAction(lambda t: {
    "tipo": "CREATE_TABLE",
    "tabla": t.table,
    "origen": "FILE",
    "archivo": t.filename,
    "delimitador": t.delimiter
})

# Opción B: CREATE TABLE <table_name> ( <column_defs> )
LPAR, RPAR = map(Suppress, "()")
create_table_columns = (
    CREATE.suppress()
    + TABLE.suppress()
    + table_name
    + LPAR
    + column_defs("cols")
    + RPAR
).setParseAction(lambda t: {
    "tipo": "CREATE_TABLE",
    "tabla": t.table,
    "origen": "COLUMNS",
    "columnas": [(c.col_name, c.col_type) for c in t.cols]
})

create_table_stmt = (create_table_from_file | create_table_columns)


# 3.3 CREATE INDEX: CREATE INDEX <index_name> ON <table_name> USING <index>(<row_name>);
INDEX = CaselessKeyword("INDEX")
ON    = CaselessKeyword("ON")
USING = CaselessKeyword("USING")

create_index_stmt = (
    CREATE.suppress()
    + INDEX.suppress()
    + index_name
    + ON.suppress()
    + table_name
    + USING.suppress()
    + index_type("index_type")
    + LPAR
    + row_name("column")
    + RPAR
).setParseAction(lambda t: {
    "tipo": "CREATE_INDEX",
    "indice": t.index,
    "tabla": t.table,
    "metodo": t.index_type,
    "columna": t.column
})


# 3.4 INSERT INTO <table_name> VALUES (<values>);
INSERT = CaselessKeyword("INSERT")
INTO   = CaselessKeyword("INTO")
VALUES = CaselessKeyword("VALUES")

insert_stmt = (
    INSERT.suppress()
    + INTO.suppress()
    + table_name
    + VALUES.suppress()
    + LPAR
    + values_list
    + RPAR
).setParseAction(lambda t: {
    "tipo": "INSERT",
    "tabla": t.table,
    "valores": t.values.asList()
})


# 3.5 DELETE FROM <table_name> WHERE <condition>;
DELETE = CaselessKeyword("DELETE")

delete_stmt = (
    DELETE.suppress()
    + FROM.suppress()
    + table_name
    + WHERE.suppress()
    + condition("cond")
).setParseAction(lambda t: {
    "tipo": "DELETE",
    "tabla": t.table,
    "where": t.cond.asList()
})


# 4. Unión de todas las sentencias en un solo parser
# -------------------------------------------------
statement = (
    select_stmt 
    | create_table_stmt 
    | create_index_stmt 
    | insert_stmt 
    | delete_stmt
)

# Cada sentencia debe terminar con punto y coma (‘;’)
semicolon = Suppress(";")
full_statement = statement + semicolon

# Si queremos permitir **varias** sentencias en secuencia:
statements_list = OneOrMore(full_statement)

# 5. Ejemplos de uso
# -------------------
if __name__ == "__main__":
    ejemplos = [
        "SELECT a,b,c FROM empleados WHERE edad = 30 LIMIT 10;",
        "SELECT * FROM productos;",
        "SELECT a FROM tablaxd;",
        "CREATE TABLE clientes FROM FILE 'clientes.csv' DELIMITED ',';",
        "CREATE TABLE facturas (id INT, total FLOAT, cliente_id INT);",
        "CREATE INDEX idx_edad ON empleados USING SPIMI(edad);",
        "INSERT INTO empleados VALUES (1, 'Juan', 28);",
        "DELETE FROM productos WHERE nombre @@ 'manzana';",
    ]

    for stmt in ejemplos:
        try:
            resultado = full_statement.parseString(stmt, parseAll=True)[0]
            print(f"\nEntrada: {stmt}")
            print("Parse result:", resultado)
        except Exception as e:
            print(f"Error parseando «{stmt}»: {e}")
        