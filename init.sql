CREATE TABLE IF NOT EXISTS persons (
    id SERIAL PRIMARY KEY,
    image_path TEXT,
    name TEXT,
    surname TEXT,
    age INT,
    nationality TEXT,
    criminal_record TEXT
);