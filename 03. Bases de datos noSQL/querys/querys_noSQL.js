use clases
db.movies

/*Ejercicio 1*/
db.movies.find();

/*Ejercicio 2*/
db.movies.find().count();

/*Ejercicio 3*/
db.movies.insertOne({"title": "test", "year": 2024, "cast": ["Actor1", "Actor2"], "genres": ["Comedy"]});

/*Ejercicio 4*/
db.movies.deleteOne({"title": "test", "year": 2024});

/*Ejercicio 5*/
db.movies.find({ cast: "and" }).count();

/*Ejercicio 6*/
db.movies.updateMany({ cast: "and" }, { $pull: { cast: "and" } });

/*Ejercicio 7*/
db.movies.find({ cast: [] }).count();

/*Ejercicio 8*/
db.movies.updateMany({ cast: [] }, {$set: {"cast": ["Undefined"]} });

/*Ejercicio 9*/
db.movies.find({ genres: [] }).count();

/*Ejercicio 10*/
db.movies.updateMany({ genres: [] }, {$set: {"genres": ["Undefined"]} });

/*Ejercicio 11*/
db.movies.find({}, {year: 1, "_id": 0}).sort({year: -1}).limit(1);

/*Ejercicio 12*/
db.movies.aggregate([
    { // Encontrar el anho mas reciente
        $group: {
            _id: null,
            maxYear: { $max: "$year" }
        }
    },
    { // Encontrar el inicio del intervalo
        $addFields: {
            minYear: { $subtract: ["$maxYear", 20] }
        }
    },
    { // Obtenemos las peliculas en el intervalo
        $lookup: { 
            from: "movies",
            let: { minYear: "$minYear", maxYear: "$maxYear" }, // "Pasamos" las variables al lookup
            pipeline: [ 
                { 
                    $match: { 
                        $expr: {
                            $and: [
                                { $gt: ["$year", "$$minYear"] },
                                { $lte: ["$year", "$$maxYear"] }
                            ]
                        }
                    } 
                }
            ],
            as: "peliculasIntervalo" 
        } 
    },
    { // Anhadimos un campo con el total de las peliculas
        $addFields: {
            total: { $size: "$peliculasIntervalo" }
        }
    },
    { //Seleccionamos solo los campos de interes
        $project: {
            _id: 1,
            total: 1
        }
    }
]);

/*Ejercicio 13*/
db.movies.aggregate([
    { // Obtenemos las peliculas en el intervalo
        $match: {
            year: { $gte: 1960, $lte: 1969 }
        }
    },
    { // Contamos el total de peliculas
        $group: {
            _id: null,
            total: { $sum: 1 }
        }
    }
]);

/*Ejercicio 14*/
db.movies.aggregate([
    { // Agrupamos por anho y contamos las peliculas por anho
        $group: {
            _id: "$year",
            pelis: { $sum: 1 }
        }
    },
    { // Encontramos el maximo
        $group: {
            _id: null,
            maxPelis: { $max: "$pelis" },
            years: 
            {  // Guardamos los anhos y su total de peliculas
                $push: {
                    year: "$_id",
                    pelis: "$pelis"
                }
            }
        }
    },
    { // Obtenemos un documento por anho
        $unwind: "$years"
    },
    { // Filtramos por el anho con el maximo de peliculas
        $match: { 
            $expr: { $eq: ["$years.pelis", "$maxPelis"] }
        }
    },
    { // Ajustamos el reesultado
        $project: {
            _id: "$years.year",
            pelis: "$years.pelis"
        }
    }
]);

/*Ejercicio 15*/
db.movies.aggregate([
    { // Agrupamos por anho y contamos las peliculas por anho
        $group: {
            _id: "$year",
            pelis: { $sum: 1 }
        }
    },
    { // Encontramos el minimo
        $group: {
            _id: null,
            minPelis: { $min: "$pelis" },
            years: 
            {  // Guardamos los anhos y su total de peliculas
                $push: { year: "$_id", pelis: "$pelis" }
            }
        }
    },
    { // Obtenemos un documento por anho
        $unwind: "$years"
    },
    { // Filtramos por el anho con el minimo de peliculas
        $match: { $expr: { $eq: ["$years.pelis", "$minPelis"] } }
    },
    { // Ajustamos el reesultado
        $project: {
            _id: "$years.year",
            pelis: "$years.pelis"
        }
    }
]);

/*Ejercicio 16*/
db.movies.aggregate([
    { // Hacemos el unwind
        $unwind: "$cast"
    },
    { // Eliminamos el id
        $project: { _id: 0 }
    },
    { // Guardamos en la coleccion actors
        $out: "actors"
    }
]);
db.actors.find().count();

/*Ejercicio 17*/
db.actors.aggregate([
    { // Excluimos los actores llamados Undefined
        $match: {
            cast: { $ne: "Undefined" }
        }
    },
    { // Agrupamos por actor y contamos sus apariciones
        $group: {
            _id: "$cast",
            cuenta: { $sum: 1 }
        }
    },
    { // Ordenamos por el total de apariciones de forma descendente
        $sort: { cuenta: -1 }
    },
    { // Limitamos a 5 el output
        $limit: 5
    }
]);

/*Ejercicio 18*/
db.actors.aggregate([
    { // Agrupamos por pelicula y anho y contamos el total de actores
        $group: {
            _id: { title: "$title", year: "$year" },
            cuenta: { $sum: 1 }
        }
    },
    { // Ordenamos por el total de actores de forma descendente
        $sort: { cuenta: -1 }
    },
    { // Limitamos a 5 el output
        $limit: 5
    }
]);

/*Ejercicio 19*/
db.actors.aggregate([
    { // Excluimos los actores undefined
        $match: {
            cast: { $ne: "Undefined" }
        }
    },
    { // Agrupamos por actor y hallamos la fecha de inicio y fin
        $group: {
            _id: "$cast",
            comienza: { $min: "$year" },
            termina: { $max: "$year" }
        }
    },
    { // Anhadimos un campo con los anhos trabajados
        $addFields: { anos: {$subtract: ["$termina", "$comienza"]} }
    },
    { // Ordenamos por la duracion descendente
        $sort: { anos: -1 }
    },
    { // Nos quedamos con el maximo
        $limit: 5
    }
]);

/*Ejercicio 20*/
db.actors.aggregate([
    { // Hacemos el unwind
        $unwind: "$genres"
    },
    { // Eliminamos el id
        $project: { _id: 0 }
    },
    { // Guardamos en la coleccion genres
        $out: "genres"
    }
]);
db.genres.count();

/*Ejercicio 21*/
db.genres.aggregate([
    { // Agrupamos por anho y genero y hallamos las peliculas distintas
        $group: {
            _id: { year: "$year", genre: "$genres" },
            pelisUnicas: { $addToSet: "$title" }
        }
    },
    { // Anhadimos el contador de pelis unicas
        $project:{
            _id: 1,
            pelis: { $size: "$pelisUnicas" }
        }
    },
    { // Ordenamos por el contador descendente
        $sort: { pelis: -1 }
    },
    { // Nos quedamos con los cinco primeros
        $limit: 5
    }
]);

/*Ejercicio 22*/
db.genres.aggregate([
    { // Eliminamos los actores "Undefined"
        $match: {
            cast: { $ne: "Undefined" }
        }
    },
    { // Agrupamos por actor y hallamos los generos distintos
        $group: {
            _id: "$cast",
            generos: { $addToSet: "$genres" }
        }
    },
    { // Anhadimos el numero de generos
        $addFields: { numgeneros: { $size: "$generos" } }
    },
    { // Ordenamos por el numero de generos descendente
        $sort: { numgeneros: -1 }
    },
    { // Nos quedamos con los cinco primeros
        $limit: 5
    }
]);

/*Ejercicio 23*/
db.genres.aggregate([
    { // Agrupamos por titulo y anho y hallamos los generos distintos
        $group: {
            _id: { title: "$title", year: "$year" },
            generos: { $addToSet: "$genres" }
        }
    },
    { // Anhadimos el numero de generos
        $addFields: { numgeneros: { $size: "$generos" } }
    },
    { // Ordenamos por el numero de generos descendente
        $sort: { numgeneros: -1 }
    },
    { // Nos quedamos con los cinco primeros
        $limit: 5
    }
]);

/*Ejercicio 24 - Pelicula con el reparto mas grande */
db.movies.aggregate([
    { // Anhadimos un campo con el tamanho del cast
        $addFields: { castSize: { $size: "$cast" } }
    },
    { // Ordenamos por el tamanho del cast
        $sort: { castSize: -1 }
    },
    { // Limitamos a uno el resultado
        $limit: 1
    },
    { // Eliminamos el id del output y los generos
        $project: { _id: 0, genres: 0 }
    }
]);


/*Ejercicio 25 - Actor mas popular por genero */
db.actors.aggregate([
    { // Excluimos los actores y generos Undefined
        $match: {
            $and: [ { genres: { $ne: "Undefined"} }, { cast: { $ne: "Undefined"} } ]
        }
    },
    { // Hacemos un unwind de los generos
        $unwind: "$genres"
    },
    { // Agrupamos por genero y actor y contamos
        $group: {
            _id: { genero: "$genres", actor: "$cast" },
            totalPeliculas: { $sum: 1 }
        }
    },
    { // Ordenamos por genero y total de peliculas
        $sort: { "_id.genre": 1, totalPeliculas: -1 }
    },
    { // Agrupamos por genero y seleccionamos el actor con mas peliculas
        $group: {
            _id: "$_id.genero",
            topActor: { $first: "$_id.actor" },
            totalPeliculas: { $first: "$totalPeliculas" }
        }
    },
    {
        $project: {
            _id: 0,
            genero: "$_id",
            topActor: 1,
            totalPeliculas: 1
        }
    }
]);

/*Ejercicio 26 - Genero mas popular por decada */
db.genres.aggregate([
    { // Excluimos los generos que sean Undefined
        $match: { genres: { $ne: "Undefined"} }
    },
    { // Calculamos la decada, usando la funcion $mod
        $addFields: {
            decada: { $subtract: ["$year", { $mod: [ "$year", 10 ] } ] }
        }
    },
    { // Agrupamos por decada y genero y calculamos el total
        $group: {
            _id: { decada: "$decada", genero: "$genres" },
            total: { $sum: 1 }
        }
    },
    { // Ordenamos por decada, de forma ascendente, y luego por el total de generos
        $sort: { "_id.decada": 1, total: -1 }
    },
    { // Agrupamos por decada para obtener el genero mas popular, y el conteo de peliculas
        $group: {
            _id: "$_id.decada",
            generoMasPopular: { $first: "$_id.genero" },
            totalPeliculas: { $first: "$total" }
        }    
    },
    { // Ordenamos por decada de manera ascendente
        $sort: { "_id": 1}
    }
]);