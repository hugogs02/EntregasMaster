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
(4, 1, 'Actuación musical de Sabela'),
(3, 1, 'Actuación musical de Rosalía'),
(3, 1, 'Actuación musical de los teloneros de Rosalía');

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
(3,6,127.4), 
(4,6,156.6), 
(4,4,73.91), 
(5,1,125.4), 
(6,2,200.5); 

/* ---------------------------------------------------------
Consultas, modificaciones, borrados y vistas
---------------------------------------------------------- */
set sql_safe_updates=0;

-- Comprobamos el funcionamiento del trigger
-- INSERT INTO AcudeA (email, telefono, id_evento, valoracion) VALUES ('carlosml@gmail.com', '634646665' , 4, 10);

DROP VIEW IF EXISTS ValoracionesEventos;
CREATE VIEW ValoracionesEventos AS 
SELECT e.id_evento, e.descripcion as evento, l.nombre as localizacion, avg(aa.valoracion) as valoracion, count(aa.email) as asistencia
FROM Evento e 
JOIN Localizacion l 
    ON e.id_localizacion = l.id_localizacion
LEFT JOIN AcudeA aa 
    ON e.id_evento = aa.id_evento
GROUP BY e.id_evento, e.descripcion, l.nombre
;

DROP VIEW IF EXISTS GananciasEvento;
CREATE VIEW GananciasEvento AS
SELECT e.descripcion AS evento,
((e.precio_entrada * (SELECT count(email) from AcudeA aa where aa.id_evento = e.id_evento)) - (select sum(coste) from Actividad a where a.id_evento = e.id_evento) - l.alquiler) as ganancia
FROM Evento e
JOIN localizacion l 
	ON e.id_localizacion = l.id_localizacion
;

-- Cambiamos el precio de la entrada del evento 'Concierto de Sabela Rodríguez'
UPDATE Evento set precio_entrada=60 where id_evento=4;

-- Consulta 1: eventos por ciudad
SELECT l.ciudad, count(e.descripcion)
FROM Evento e
JOIN Localizacion l
	ON e.id_localizacion = l.id_localizacion
GROUP BY l.ciudad
;

-- Consulta 2: Actividades que superan el coste promedio
SELECT a.nombre, a.coste
FROM Actividad a
WHERE a.coste >= (SELECT AVG(coste) FROM Actividad)
;

-- Consulta 3: Ganancias de los artistas
SELECT a.nombre AS Artista, SUM(act.cache) AS Ganancias
FROM Artista a
JOIN ActuaEn act
	ON a.id_artista = act.id_artista
GROUP BY a.nombre
ORDER BY Ganancias DESC;

-- Consulta 4: Eventos menos populares
WITH Asistencia AS (
	SELECT e.descripcion, COUNT(a.email) as asistentes, 
    ROW_NUMBER() OVER (ORDER BY COUNT(a.email) ASC) AS ranking
    FROM Evento e
    JOIN AcudeA a
		ON e.id_evento = a.id_evento
	GROUP BY e.descripcion
)
SELECT descripcion, asistentes FROM Asistencia
where ranking <=2
;

-- Consulta 5: Valoración promedio
SELECT p.nombre, p.email, p.telefono, ROUND(AVG(a.valoracion), 2) as "Valoración promedio"
FROM Persona p
JOIN AcudeA a
	ON p.email = a.email
    AND p.telefono = a.telefono
GROUP BY p.nombre, p.email, p.telefono
;

-- Consulta 6: Eventos con más actividades
SELECT e.descripcion AS Evento, l.nombre as Localizacion, COUNT(a.id_actividad) as Actividades
FROM Evento e
JOIN Localizacion l
	ON e.id_localizacion = l.id_localizacion
JOIN Actividad a
	ON a.id_evento = e.id_evento
GROUP BY Evento, Localizacion
ORDER BY Actividades DESC
LIMIT 3
;

-- Consulta 7: Eventos más costosos
SELECT e.descripcion as Evento, l.nombre as Localizacion, SUM(a.coste) as Coste
FROM Evento e
JOIN Localizacion l
	ON e.id_localizacion = l.id_localizacion
JOIN Actividad a
	ON a.id_evento = e.id_evento
GROUP BY Evento, Localizacion
ORDER BY Coste DESC
;

-- Consulta 8: Porcentaje de ocupación de cada evento
SELECT e.descripcion as Evento, round(((COUNT(a.email)/l.aforo)*100), 2) as Ocupacion
FROM Evento e
JOIN Localizacion l
	ON e.id_localizacion = l.id_localizacion
JOIN AcudeA a
	ON a.id_evento = e.id_evento
GROUP BY Evento, l.id_localizacion
;