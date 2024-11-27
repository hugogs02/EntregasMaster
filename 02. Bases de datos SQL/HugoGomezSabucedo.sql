/* -------------------------------------------------------
Hugo Gomez Sabucedo
ArteVida Cultural
-------------------------------------------------------- */
/* ---------------------------------------------------------
Definicion de la estructura de la base de datos 
---------------------------------------------------------- */
DROP DATABASE IF EXISTS ArteVidaCultural;

CREATE DATABASE ArteVidaCultural;

use ArteVidaCultural;

DROP TABLE IF EXISTS Localizacion;
CREATE TABLE Localizacion(
    id_localizacion INT AUTO_INCREMENT,
    nombre VARCHAR(75),
    direccion VARCHAR(100),
    ciudad VARCHAR(50),
    aforo INT,
    alquiler DECIMAL(10,2),
    caracteristicas TEXT,

    PRIMARY KEY (id_localizacion)
);

DROP TABLE IF EXISTS Evento;
CREATE TABLE Evento(
    id_evento INT AUTO_INCREMENT,
    id_localizacion INT,
    fecha TIMESTAMP,
    precio_entrada DECIMAL (10,2),
    descripcion VARCHAR(500),

    PRIMARY KEY (id_evento),
    FOREIGN KEY (id_localizacion) REFERENCES Localizacion(id_localizacion)
    ON DELETE CASCADE ON UPDATE CASCADE
);

DROP TABLE IF EXISTS TipoActividad;
CREATE TABLE TipoActividad(
    id_tipo INT AUTO_INCREMENT,
    tipo VARCHAR(30),
    descripcion VARCHAR(100),

    PRIMARY KEY (id_tipo)
);

DROP TABLE IF EXISTS Actividad;
CREATE TABLE Actividad(
    id_actividad INT AUTO_INCREMENT,
    id_evento INT,
    id_tipo INT,
    nombre VARCHAR(50),
    coste DECIMAL(10,2) DEFAULT 0,

    PRIMARY KEY (id_actividad),
    FOREIGN KEY (id_evento) REFERENCES Evento(id_evento)
    ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (id_tipo) REFERENCES TipoActividad(id_tipo)
    ON DELETE SET NULL ON UPDATE CASCADE
);

DROP TABLE IF EXISTS Artista;
CREATE TABLE Artista(
    id_artista INT AUTO_INCREMENT,
    nombre VARCHAR(50),
    biografia TEXT,

    PRIMARY KEY (id_artista)
);

DROP TABLE IF EXISTS Persona;
CREATE TABLE Persona(
    email VARCHAR(75),
    telefono CHAR(9),
    nombre VARCHAR(25),
    apellido1 VARCHAR(40),
    apellido2 VARCHAR(40),

    PRIMARY KEY (email, telefono)
);

DROP TABLE IF EXISTS ActuaEn;
CREATE TABLE ActuaEn(
    id_artista INT,
    id_actividad INT,
    cache DECIMAL(10,2),

    PRIMARY KEY (id_artista, id_actividad),
    FOREIGN KEY (id_artista) REFERENCES Artista(id_artista)
    ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (id_actividad) REFERENCES Actividad(id_actividad)
    ON UPDATE CASCADE ON DELETE CASCADE
);

DROP TABLE IF EXISTS AcudeA;
CREATE TABLE AcudeA(
    email VARCHAR(75),
    telefono CHAR(9),
    id_evento INT,
    valoracion INT,

    PRIMARY KEY (email, telefono, id_evento),
    FOREIGN KEY (email, telefono) REFERENCES Persona(email, telefono)
    ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (id_evento) REFERENCES Evento(id_evento)
    ON DELETE CASCADE ON UPDATE CASCADE
);

/* ---------------------------------------------------------
Trigger
Insercion de datos
---------------------------------------------------------- */

DROP TRIGGER IF EXISTS actualiza_coste_actividad_insert;

DELIMITER $$
CREATE TRIGGER actualiza_coste_actividad_insert
AFTER INSERT ON ActuaEn
FOR EACH ROW
BEGIN
    UPDATE Actividad
    SET coste = (
        SELECT COALESCE(SUM(cache), 0)
        FROM ActuaEn
        WHERE id_actividad = NEW.id_actividad
    )
    WHERE id_actividad = NEW.id_actividad;
END;
$$
DELIMITER ;

DROP TRIGGER IF EXISTS actualiza_coste_actividad_update;

DELIMITER $$
CREATE TRIGGER actualiza_coste_actividad_update
AFTER UPDATE ON ActuaEn
FOR EACH ROW
BEGIN
    UPDATE Actividad
    SET coste = (
        SELECT COALESCE(SUM(cache), 0)
        FROM ActuaEn
        WHERE id_actividad = NEW.id_actividad
    )
    WHERE id_actividad = NEW.id_actividad;
END;
$$
DELIMITER ;

DROP TRIGGER IF EXISTS valida_aforo;

DELIMITER $$
CREATE TRIGGER valida_aforo
BEFORE INSERT ON AcudeA
FOR EACH ROW
BEGIN
	DECLARE AforoMax INT;
    DECLARE AforoAct INT; 
    
    SELECT l.aforo
    INTO AforoMax
    FROM Evento e
    JOIN Localizacion l
        ON e.id_localizacion = l.id_localizacion
    WHERE e.id_evento = NEW.id_evento;

    SELECT COUNT(*)
    INTO AforoAct
    FROM AcudeA
    WHERE id_evento = NEW.id_evento;

    IF AforoAct >= AforoMax THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'No se pueden registrar mas asistentes que el aforo maximo del recinto.';
    END IF;
END;
$$
DELIMITER ;

/* -- Insercion de datos -- */
INSERT INTO Localizacion (nombre, direccion, ciudad, aforo, alquiler, caracteristicas) VALUES 
('Teatro Principal', 'rúa Nova 21', 'Santiago de Compostela', 50, 1500.00, 'Teatro construido en 1841 que acoge diferentes espectáculos'),
('Auditorio Mar de Vigo', 'avenida da Beiramar 29', 'Vigo', 1500, 5000.00, 'Palacio de Congresos con un gran auditorio situado enfrente del puerto'),
('Coliseum', 'rúa Francisco Pérez Carballo', 'A Coruña', 11000, 26000.00, 'Estadio de uso para conciertos y espectáculos y partidos de baloncesto'),
('Sala Capitol', 'rúa Concepción Arenal 5', 'Santiago de Compostela', 3, 160.00, 'Sala de conciertos (simulación)');

INSERT INTO Evento (id_localizacion, fecha, precio_entrada, descripcion) VALUES 
(1, '2024-12-01 20:00:00', 35.00, 'Concierto de música clásica'),
(2, '2024-12-10 18:00:00', 25.00, 'Conferencia sobre arte contemporáneo'),
(3, '2025-06-30 21:45:00', 75.00, 'Concierto de Rosalía'),
(4, '2024-12-15 21:00:00', 50.00, 'Concierto de Sabela Rodríguez');

INSERT INTO TipoActividad (tipo, descripcion) VALUES 
('Concierto', 'Concierto musical de un artista'),
('Conferencia', 'Charla o exposición sobre diversos temas'),
('Teatro', 'Representación de una obra dramática o comedia');

INSERT INTO Actividad (id_evento, id_tipo, nombre) VALUES 
(1, 1, 'Concierto de piezas de Beethoven'),
(1, 1, 'Concierto de piezas de Mozart'),
(2, 2, 'Arte en el Siglo XXI'),
(3, 1, 'Actuación musical de Sabela'),
(4, 1, 'Actuación musical de Rosalía'),
(4, 1, 'Actuación musical de los teloneros de Rosalía');

INSERT INTO Artista (nombre, biografia) VALUES 
('Rosalía', 'La cantante española más conocida'),
('Ana Mena', 'Cantante española de pop'),
('Sabela Rodríguez', 'Cantante gallega con temas en gallego'),
('Tanxugueiras', 'Trío de cantantes gallegas'),
('Juan López', 'Actor de teatro con una amplia trayectoria en obras clásicas'),
('Carlos Ruiz', 'Pianista concertista, especializado en música barroca');

INSERT INTO Persona (email, telefono, nombre, apellido1, apellido2) VALUES 
('anagomez@gmailcom', '655316631', 'Ana', 'Gómez', 'Rodríguez'),
('lupesa@gmail.com', '726654546', 'Luis', 'Pérez', 'Sánchez'),
('carlosml@gmail.com', '634646665', 'Carlos', 'Martín', 'Lopez'),
('hugogosa@gmail.com', '695321467', 'Hugo', 'Gómez', 'Sabucedo');

INSERT INTO AcudeA (email, telefono, id_evento, valoracion) VALUES 
('anagomez@gmailcom', '655316631', 1, 7), 
('anagomez@gmailcom', '655316631', 2, 8),
('hugogosa@gmail.com', '695321467', 2, 7),
('hugogosa@gmail.com', '695321467', 3, 10),
('anagomez@gmailcom', '655316631', 4, 9),
('hugogosa@gmail.com', '695321467', 4, 7),
('lupesa@gmail.com', '726654546', 4, 5);

INSERT INTO ActuaEn (id_artista, id_actividad, cache) VALUES
(1,5,1500.20), 
(2,6,987.65), 
(3,4,127.4), 
(4,4,156.6), 
(4,6,73.91), 
(5,1,125.4), 
(6,2,200.5); 

/* ---------------------------------------------------------
Consultas, modificaciones, borrados y vistas
---------------------------------------------------------- */
set sql_safe_updates=0;

-- Comprobamos el funcionamiento del trigger
-- INSERT INTO AcudeA (email, telefono, id_evento, valoracion) VALUES ('carlosml@gmail.com', '634646665' , 4, 10);

CREATE VIEW ValoracionesEventos AS 
SELECT e.id_evento, e.descripcion as evento, l.nombre as localizacion, avg(aa.valoracion) as valoracion, count(aa.email) as asistencia
FROM Evento e 
JOIN Localizacion l 
    ON e.id_localizacion = l.id_localizacion
LEFT JOIN AcudeA aa 
    ON e.id_evento = aa.id_evento
GROUP BY e.id_evento, e.descripcion, l.nombre
;
