"""Construye data/{train,val,test}.jsonl a partir de ejemplos escritos a mano.

DATOS SINTÉTICOS: todas las preguntas y respuestas se refieren a un reglamento
FICTICIO (data/reglamento_ficticio.md). Fueron redactadas con asistencia de IA
para este ejercicio académico; no contienen datos personales ni confidenciales.

Cada fila: (split, tipo, articulos, pregunta, respuesta, hechos_clave)
  tipo: "simple" (un artículo), "combinada" (dos artículos), "fuera_alcance"
  hechos_clave: fragmentos que DEBEN aparecer en una respuesta correcta.
"""

import json
from pathlib import Path

NO = "El reglamento del LCA no regula ese aspecto; consulta directamente con la coordinación del laboratorio."

E = [
    # ---------------- TRAIN: simples (88) ----------------
    # Art. 1
    ("train", "simple", [1], "¿Qué regula el reglamento del LCA?", "Regula el uso del Laboratorio de Cómputo Acelerado \"Nodo Cóndor\" (LCA). (Art. 1)", ["Nodo Cóndor"]),
    ("train", "simple", [1], "¿Cuántas GPUs tiene en total el Nodo Cóndor?", "El LCA tiene 24 GPUs en total, repartidas en 6 servidores de 4 GPUs cada uno. (Art. 1)", ["24 GPUs"]),
    ("train", "simple", [1], "¿Cuántos servidores componen el laboratorio?", "El LCA cuenta con 6 servidores, cada uno con 4 GPUs. (Art. 1)", ["6 servidores", "4 GPUs"]),
    ("train", "simple", [1], "Describe la infraestructura de cómputo del LCA.", "El LCA dispone de 6 servidores de 4 GPUs cada uno, para un total de 24 GPUs. (Art. 1)", ["6 servidores", "24 GPUs"]),
    # Art. 2
    ("train", "simple", [2], "¿Quiénes pueden usar el laboratorio?", "Pueden usarlo los estudiantes de pregrado desde quinto semestre, los estudiantes de posgrado y los docentes. (Art. 2)", ["quinto semestre", "posgrado", "docentes"]),
    ("train", "simple", [2], "Estoy en tercer semestre de pregrado, ¿puedo usar el LCA?", "No. Los estudiantes de pregrado pueden usar el LCA solo desde quinto semestre. (Art. 2)", ["quinto semestre"]),
    ("train", "simple", [2], "¿Puede usar el laboratorio una persona de otra institución?", "Solo si existe un convenio vigente con su institución; los usuarios externos no tienen acceso sin convenio. (Art. 2)", ["convenio vigente"]),
    ("train", "simple", [2], "¿Desde qué semestre puede un estudiante de pregrado acceder al LCA?", "Desde quinto semestre. (Art. 2)", ["quinto semestre"]),
    # Art. 3
    ("train", "simple", [3], "¿Cómo solicito acceso al laboratorio?", "Debes diligenciar el formulario LCA-01 firmado por un docente tutor. (Art. 3)", ["LCA-01", "docente tutor"]),
    ("train", "simple", [3], "¿Cuánto tarda la respuesta a una solicitud de acceso?", "La coordinación responde en un plazo máximo de 3 días hábiles. (Art. 3)", ["3 días hábiles"]),
    ("train", "simple", [3], "¿Quién debe firmar el formulario de acceso?", "El formulario LCA-01 debe estar firmado por un docente tutor. (Art. 3)", ["docente tutor"]),
    ("train", "simple", [3], "¿Qué formulario se usa para pedir acceso al LCA?", "Se usa el formulario LCA-01. (Art. 3)", ["LCA-01"]),
    # Art. 4
    ("train", "simple", [4], "¿Tengo que hacer algún curso antes de usar el laboratorio?", "Sí, debes aprobar el curso de inducción \"Cóndor Básico\", de 4 horas. (Art. 4)", ["Cóndor Básico", "4 horas"]),
    ("train", "simple", [4], "¿Qué nota necesito para aprobar la inducción?", "Necesitas una nota mínima de 80 sobre 100 en el curso \"Cóndor Básico\". (Art. 4)", ["80 sobre 100"]),
    ("train", "simple", [4], "¿Cuánto tiempo es válida la certificación de inducción?", "La certificación del curso \"Cóndor Básico\" tiene una vigencia de 12 meses. (Art. 4)", ["12 meses"]),
    ("train", "simple", [4], "¿Cuánto dura el curso Cóndor Básico?", "El curso \"Cóndor Básico\" dura 4 horas. (Art. 4)", ["4 horas"]),
    # Art. 5
    ("train", "simple", [5], "¿Puedo prestarle mi cuenta a un compañero?", "No. Cada cuenta es personal e intransferible. (Art. 5)", ["personal e intransferible"]),
    ("train", "simple", [5], "¿Cada cuánto debo cambiar la contraseña?", "Debes cambiar la contraseña cada 90 días. (Art. 5)", ["90 días"]),
    ("train", "simple", [5], "¿Es obligatoria la autenticación de dos factores?", "Sí, la autenticación de dos factores es obligatoria para todas las cuentas. (Art. 5)", ["dos factores"]),
    ("train", "simple", [5], "¿Qué reglas de seguridad aplican a las cuentas del LCA?", "Las cuentas son personales e intransferibles, la contraseña se cambia cada 90 días y la autenticación de dos factores es obligatoria. (Art. 5)", ["personal", "90 días", "dos factores"]),
    # Art. 6
    ("train", "simple", [6], "¿En qué horario abre la sala presencial entre semana?", "De lunes a viernes, de 7:00 a 21:00. (Art. 6)", ["7:00", "21:00"]),
    ("train", "simple", [6], "¿El laboratorio abre los sábados?", "Sí, los sábados la sala presencial abre de 8:00 a 13:00. (Art. 6)", ["8:00", "13:00"]),
    ("train", "simple", [6], "¿Puedo conectarme de forma remota a medianoche?", "Sí, el acceso remoto está disponible las 24 horas, todos los días. (Art. 6)", ["24 horas"]),
    ("train", "simple", [6], "¿Cuál es el horario del LCA?", "La sala presencial abre de lunes a viernes de 7:00 a 21:00 y los sábados de 8:00 a 13:00; el acceso remoto funciona las 24 horas. (Art. 6)", ["7:00", "21:00", "8:00", "13:00", "24 horas"]),
    # Art. 7
    ("train", "simple", [7], "¿Dónde reservo una GPU?", "Las GPUs se reservan en el portal \"Agenda Cóndor\". (Art. 7)", ["Agenda Cóndor"]),
    ("train", "simple", [7], "¿Con cuánta anticipación debo reservar?", "Debes reservar con al menos 48 horas de anticipación. (Art. 7)", ["48 horas"]),
    ("train", "simple", [7], "¿Cuánto puede durar un bloque de reserva?", "Cada bloque de reserva dura como máximo 8 horas continuas. (Art. 7)", ["8 horas"]),
    ("train", "simple", [7], "¿Puedo reservar una GPU para mañana mismo?", "No, las reservas en \"Agenda Cóndor\" deben hacerse con al menos 48 horas de anticipación. (Art. 7)", ["48 horas"]),
    # Art. 8
    ("train", "simple", [8], "¿Cuántas GPUs puede usar a la vez un estudiante de pregrado?", "Un estudiante de pregrado puede usar como máximo 2 GPUs simultáneas. (Art. 8)", ["2 GPUs"]),
    ("train", "simple", [8], "¿Cuál es el máximo de GPUs simultáneas para posgrado?", "Un estudiante de posgrado puede usar como máximo 4 GPUs simultáneas. (Art. 8)", ["4 GPUs"]),
    ("train", "simple", [8], "¿Cuántas GPUs puede ocupar un docente al mismo tiempo?", "Un docente puede usar como máximo 8 GPUs simultáneas. (Art. 8)", ["8 GPUs"]),
    ("train", "simple", [8], "¿Cuáles son los cupos simultáneos de GPU según el tipo de usuario?", "Pregrado: máximo 2 GPUs; posgrado: 4 GPUs; docentes: 8 GPUs simultáneas. (Art. 8)", ["2 GPUs", "4 GPUs", "8 GPUs"]),
    # Art. 9
    ("train", "simple", [9], "¿Cuántas horas-GPU por semana tiene un estudiante de pregrado?", "Pregrado tiene una cuota de 20 horas-GPU por semana. (Art. 9)", ["20 horas-GPU"]),
    ("train", "simple", [9], "¿Cuál es la cuota semanal de posgrado?", "Posgrado tiene una cuota de 60 horas-GPU por semana. (Art. 9)", ["60 horas-GPU"]),
    ("train", "simple", [9], "¿Cuándo se reinicia la cuota semanal?", "La cuota se reinicia cada lunes a las 00:00. (Art. 9)", ["lunes", "00:00"]),
    ("train", "simple", [9], "Explícame cómo funciona la cuota semanal de GPU.", "Pregrado dispone de 20 horas-GPU y posgrado de 60 horas-GPU por semana; la cuota se reinicia cada lunes a las 00:00. (Art. 9)", ["20 horas-GPU", "60 horas-GPU", "lunes"]),
    # Art. 10
    ("train", "simple", [10], "¿Con cuánta anticipación debo cancelar una reserva?", "Debes cancelarla con al menos 6 horas de anticipación. (Art. 10)", ["6 horas"]),
    ("train", "simple", [10], "¿Qué pasa si no me presento a mis reservas?", "Si no te presentas a 3 reservas en un mismo mes, quedas suspendido para reservar durante 15 días. (Art. 10)", ["3 reservas", "15 días"]),
    ("train", "simple", [10], "¿Cuánto dura la suspensión por no presentarse a reservas?", "La suspensión para reservar dura 15 días. (Art. 10)", ["15 días"]),
    ("train", "simple", [10], "¿Cuántas inasistencias a reservas se toleran al mes?", "Al acumular 3 reservas sin presentarte en un mismo mes quedas suspendido para reservar durante 15 días. (Art. 10)", ["3 reservas", "15 días"]),
    # Art. 11
    ("train", "simple", [11], "Necesito entrenar un modelo durante 20 horas, ¿qué hago?", "Los trabajos de más de 8 horas requieren una solicitud de \"Corrida Extendida\" con el formulario LCA-07, aprobada por el coordinador. (Art. 11)", ["Corrida Extendida", "LCA-07"]),
    ("train", "simple", [11], "¿Cuál es la duración máxima de una corrida extendida?", "Una corrida extendida dura como máximo 72 horas. (Art. 11)", ["72 horas"]),
    ("train", "simple", [11], "¿Quién aprueba una corrida extendida?", "La aprueba el coordinador del laboratorio. (Art. 11)", ["coordinador"]),
    ("train", "simple", [11], "¿Qué formulario se usa para trabajos largos?", "Se usa el formulario LCA-07 de \"Corrida Extendida\". (Art. 11)", ["LCA-07"]),
    # Art. 12
    ("train", "simple", [12], "¿Quién tiene prioridad cuando dos reservas chocan?", "La prioridad es: 1) tesis con sustentación en menos de 30 días; 2) cursos; 3) proyectos de investigación; 4) uso libre. (Art. 12)", ["tesis", "cursos", "investigación", "uso libre"]),
    ("train", "simple", [12], "¿Qué uso tiene la máxima prioridad en el laboratorio?", "Las tesis con sustentación en menos de 30 días. (Art. 12)", ["tesis", "30 días"]),
    ("train", "simple", [12], "¿Qué prioridad tiene el uso libre?", "El uso libre tiene la última prioridad, después de tesis, cursos y proyectos de investigación. (Art. 12)", ["última"]),
    ("train", "simple", [12], "¿Los cursos tienen prioridad sobre los proyectos de investigación?", "Sí, los cursos son la prioridad 2 y los proyectos de investigación la prioridad 3. (Art. 12)", ["prioridad 2", "prioridad 3"]),
    # Art. 13
    ("train", "simple", [13], "¿Cuánto espacio tengo en /home?", "Cada usuario dispone de 200 GB en /home. (Art. 13)", ["200 GB"]),
    ("train", "simple", [13], "¿Cada cuánto se borra /scratch?", "El espacio /scratch se borra cada 14 días. (Art. 13)", ["14 días"]),
    ("train", "simple", [13], "¿De qué tamaño es el espacio /scratch?", "El espacio /scratch es de 1 TB compartido y es temporal. (Art. 13)", ["1 TB"]),
    ("train", "simple", [13], "¿Qué espacios de almacenamiento ofrece el LCA?", "Cada usuario tiene 200 GB en /home y existe un /scratch temporal de 1 TB compartido que se borra cada 14 días. (Art. 13)", ["200 GB", "1 TB", "14 días"]),
    # Art. 14
    ("train", "simple", [14], "¿Se hace copia de seguridad de /home?", "Sí, /home se respalda todos los días a las 2:00. (Art. 14)", ["2:00"]),
    ("train", "simple", [14], "¿Cuánto tiempo se guardan los respaldos?", "Las copias de /home se conservan 30 días. (Art. 14)", ["30 días"]),
    ("train", "simple", [14], "¿Tiene respaldo el espacio /scratch?", "No, /scratch no tiene respaldo. (Art. 14)", ["no tiene respaldo"]),
    # Art. 15
    ("train", "simple", [15], "¿Puedo subir una base de datos con nombres y cédulas de personas?", "No, está prohibido cargar datos personales sin anonimizar. (Art. 15)", ["sin anonimizar"]),
    ("train", "simple", [15], "¿Qué necesito para trabajar con datos sensibles?", "Necesitas el aval previo del Comité de Ética de Datos (CED). (Art. 15)", ["Comité de Ética de Datos"]),
    ("train", "simple", [15], "¿Qué es el CED?", "Es el Comité de Ética de Datos, que debe avalar previamente el uso de datos sensibles. (Art. 15)", ["Comité de Ética de Datos"]),
    # Art. 16
    ("train", "simple", [16], "¿Puedo usar Docker en el laboratorio?", "No, Docker no está permitido; puedes usar contenedores Apptainer. (Art. 16)", ["Docker no", "Apptainer"]),
    ("train", "simple", [16], "¿Cómo instalo mis librerías de Python?", "Solo puedes instalar software en entornos de usuario: uv, conda o contenedores Apptainer. (Art. 16)", ["uv", "conda", "Apptainer"]),
    ("train", "simple", [16], "¿Cuánto tarda la instalación de software de sistema?", "El software de sistema se solicita por ticket y se atiende en 5 días hábiles. (Art. 16)", ["ticket", "5 días hábiles"]),
    # Art. 17
    ("train", "simple", [17], "¿Puedo minar criptomonedas en las GPUs?", "No, está prohibido minar criptomonedas en el LCA. (Art. 17)", ["prohibido"]),
    ("train", "simple", [17], "¿Puedo ejecutar un proceso largo en el nodo de acceso?", "No, está prohibido ejecutar procesos de más de 10 minutos en el nodo de acceso. (Art. 17)", ["10 minutos"]),
    ("train", "simple", [17], "¿Qué actividades están prohibidas en el LCA?", "Minar criptomonedas, ejecutar videojuegos, compartir credenciales y ejecutar procesos de más de 10 minutos en el nodo de acceso. (Art. 17)", ["criptomonedas", "videojuegos", "credenciales", "10 minutos"]),
    # Art. 18
    ("train", "simple", [18], "¿Puedo almorzar en la sala de servidores?", "No, está prohibido comer en la sala de servidores. (Art. 18)", ["prohibido comer"]),
    ("train", "simple", [18], "¿Se permiten bebidas en la sala de estaciones?", "Sí, pero solo en recipientes con tapa. (Art. 18)", ["con tapa"]),
    ("train", "simple", [18], "¿Qué dice el reglamento sobre alimentos?", "Está prohibido comer en la sala de servidores y en la sala de estaciones solo se permiten bebidas con tapa. (Art. 18)", ["prohibido comer", "con tapa"]),
    # Art. 19
    ("train", "simple", [19], "Un servidor dejó de responder, ¿dónde lo reporto?", "Repórtalo en la mesa de ayuda \"MesaCóndor\", indicando el código del nodo afectado. (Art. 19)", ["MesaCóndor", "código del nodo"]),
    ("train", "simple", [19], "¿En cuánto tiempo debo reportar una falla?", "En un plazo máximo de 2 horas. (Art. 19)", ["2 horas"]),
    ("train", "simple", [19], "¿Qué información debo incluir al reportar un incidente?", "Debes indicar el código del nodo afectado al reportar en \"MesaCóndor\". (Art. 19)", ["código del nodo"]),
    # Art. 20
    ("train", "simple", [20], "¿Cuándo se hace el mantenimiento del laboratorio?", "El primer miércoles de cada mes, de 6:00 a 12:00. (Art. 20)", ["primer miércoles", "6:00", "12:00"]),
    ("train", "simple", [20], "¿Con cuánta anticipación avisan el mantenimiento?", "Se avisa con 7 días de anticipación. (Art. 20)", ["7 días"]),
    ("train", "simple", [20], "¿A qué hora termina el mantenimiento programado?", "Termina a las 12:00; empieza a las 6:00 del primer miércoles de cada mes. (Art. 20)", ["12:00"]),
    # Art. 21
    ("train", "simple", [21], "¿Qué sanción tiene una falta leve?", "Una falta leve se sanciona con amonestación escrita. (Art. 21)", ["amonestación escrita"]),
    ("train", "simple", [21], "¿Cuánto dura la suspensión por falta grave?", "Una falta grave se sanciona con suspensión de 30 días. (Art. 21)", ["30 días"]),
    ("train", "simple", [21], "¿Qué ocurre con una falta gravísima?", "Se sanciona con suspensión de 6 meses y reporte a la facultad. (Art. 21)", ["6 meses", "reporte a la facultad"]),
    # Art. 22
    ("train", "simple", [22], "¿Exceder la cuota es una falta grave?", "No, exceder la cuota es una falta leve. (Art. 22)", ["leve"]),
    ("train", "simple", [22], "¿Qué tipo de falta es compartir credenciales?", "Compartir credenciales es una falta grave. (Art. 22)", ["grave"]),
    ("train", "simple", [22], "¿Qué pasa si acumulo varias faltas leves?", "Tres faltas leves en un mismo semestre equivalen a una falta grave. (Art. 22)", ["Tres faltas leves", "grave"]),
    # Art. 23
    ("train", "simple", [23], "¿Puedo apelar una sanción?", "Sí, dentro de los 5 días hábiles siguientes a la notificación, ante el Comité del Laboratorio. (Art. 23)", ["5 días hábiles", "Comité del Laboratorio"]),
    ("train", "simple", [23], "¿Ante quién se presenta una apelación?", "Ante el Comité del Laboratorio. (Art. 23)", ["Comité del Laboratorio"]),
    ("train", "simple", [23], "¿En cuánto tiempo resuelven una apelación?", "El Comité del Laboratorio responde en un plazo de 10 días hábiles. (Art. 23)", ["10 días hábiles"]),
    # Art. 24
    ("train", "simple", [24], "¿Debo mencionar el laboratorio en mi artículo científico?", "Sí, debes incluir el agradecimiento \"Este trabajo usó el Nodo Cóndor (LCA)\". (Art. 24)", ["Este trabajo usó el Nodo Cóndor"]),
    ("train", "simple", [24], "¿En qué plazo debo registrar una publicación?", "Debes registrarla ante la coordinación dentro de los 30 días siguientes a su publicación. (Art. 24)", ["30 días"]),
    ("train", "simple", [24], "¿Qué frase de agradecimiento exige el reglamento?", "\"Este trabajo usó el Nodo Cóndor (LCA)\". (Art. 24)", ["Este trabajo usó el Nodo Cóndor"]),
    # Art. 25
    ("train", "simple", [25], "¿Desde cuándo rige el reglamento?", "Rige desde el 1 de febrero de 2026. (Art. 25)", ["1 de febrero de 2026"]),
    ("train", "simple", [25], "¿Cada cuánto se revisa el reglamento?", "Se revisa cada 2 años. (Art. 25)", ["2 años"]),
    ("train", "simple", [25], "¿Cuál es la vigencia del reglamento del LCA?", "Rige desde el 1 de febrero de 2026 y se revisa cada 2 años. (Art. 25)", ["1 de febrero de 2026", "2 años"]),

    # ---------------- TRAIN: combinadas (4) ----------------
    ("train", "combinada", [2, 3], "Soy estudiante de posgrado, ¿puedo usar el LCA y cómo pido acceso?", "Sí, los estudiantes de posgrado son usuarios autorizados; el acceso se pide con el formulario LCA-01 firmado por un docente tutor. (Arts. 2 y 3)", ["posgrado", "LCA-01"]),
    ("train", "combinada", [4, 5], "¿Qué debo cumplir antes de empezar y cómo protejo mi cuenta?", "Debes aprobar el curso \"Cóndor Básico\" con mínimo 80 sobre 100; tu cuenta es intransferible, cambias la contraseña cada 90 días y usas autenticación de dos factores. (Arts. 4 y 5)", ["Cóndor Básico", "90 días", "dos factores"]),
    ("train", "combinada", [10, 22], "Si no cancelo una reserva a tiempo, ¿qué tipo de falta cometo?", "Debes cancelar con al menos 6 horas de anticipación; no hacerlo es una falta leve. (Arts. 10 y 22)", ["6 horas", "leve"]),
    ("train", "combinada", [15, 21, 22], "¿Qué sanción recibe quien carga datos personales sin anonimizar?", "Está prohibido; cargar datos personales sin autorización es una falta gravísima, sancionada con suspensión de 6 meses y reporte a la facultad. (Arts. 15, 21 y 22)", ["gravísima", "6 meses"]),

    # ---------------- TRAIN: fuera de alcance (8) ----------------
    ("train", "fuera_alcance", [], "¿Dónde puedo parquear mi carro?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cuál es la clave del wifi de la universidad?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿El laboratorio presta portátiles para llevar a casa?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cuánto cuesta el almuerzo en la cafetería?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cómo solicito una beca de posgrado?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cuántas impresiones gratis tengo al mes?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Puedo llevar a mi mascota al laboratorio?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cuál es el salario de los monitores del LCA?", NO, ["no regula"]),
    # v2: 12 rechazos adicionales (temas cercanos al laboratorio y del campus) para enseñar el límite del reglamento
    ("train", "fuera_alcance", [], "¿De qué marca son las GPUs del Nodo Cóndor?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cuánto cuesta la hora de GPU para una empresa?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿El LCA ofrece certificados de horas de uso para mi hoja de vida?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Puedo hacer una fiesta de fin de semestre en la sala de estaciones?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Quién es el decano de la facultad?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Qué sistema operativo usan los servidores?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cuándo empiezan las vacaciones de mitad de año?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Me pueden recomendar un tema de tesis?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Dónde consigo el carné estudiantil?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Qué lenguaje de programación es mejor para empezar?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿El laboratorio tiene aire acondicionado?", NO, ["no regula"]),
    ("train", "fuera_alcance", [], "¿Cuántos estudiantes hay matriculados en la universidad?", NO, ["no regula"]),

    # ---------------- VAL (20) ----------------
    ("val", "simple", [1], "¿Con cuántas tarjetas gráficas cuenta el LCA?", "El LCA tiene 24 GPUs en total (6 servidores de 4 GPUs). (Art. 1)", ["24 GPUs"]),
    ("val", "simple", [3], "¿Qué trámite hago para obtener acceso al Nodo Cóndor?", "Diligencias el formulario LCA-01 con la firma de un docente tutor; la respuesta llega en máximo 3 días hábiles. (Art. 3)", ["LCA-01", "3 días hábiles"]),
    ("val", "simple", [4], "¿Cuál es el puntaje mínimo de la inducción obligatoria?", "El puntaje mínimo es 80 sobre 100 en el curso \"Cóndor Básico\". (Art. 4)", ["80 sobre 100"]),
    ("val", "simple", [6], "¿Hasta qué hora puedo estar en la sala presencial un martes?", "Hasta las 21:00; entre semana la sala abre de 7:00 a 21:00. (Art. 6)", ["21:00"]),
    ("val", "simple", [7], "¿Cuál es la duración máxima de una reserva de GPU?", "Cada bloque de reserva dura como máximo 8 horas continuas. (Art. 7)", ["8 horas"]),
    ("val", "simple", [9], "Soy de pregrado, ¿cuántas horas de GPU tengo a la semana?", "Tienes una cuota de 20 horas-GPU por semana. (Art. 9)", ["20 horas-GPU"]),
    ("val", "simple", [11], "¿Por cuánto tiempo máximo puede correr un trabajo aprobado como extendido?", "Hasta 72 horas. (Art. 11)", ["72 horas"]),
    ("val", "simple", [13], "¿Cuál es mi límite de almacenamiento personal?", "Tienes 200 GB en /home. (Art. 13)", ["200 GB"]),
    ("val", "simple", [15], "¿Necesito permiso para usar datos de salud en mi proyecto?", "Sí, los datos sensibles requieren el aval previo del Comité de Ética de Datos (CED). (Art. 15)", ["Comité de Ética de Datos"]),
    ("val", "simple", [17], "¿Puedo jugar videojuegos en las estaciones del LCA?", "No, está prohibido ejecutar videojuegos en el LCA. (Art. 17)", ["prohibido"]),
    ("val", "simple", [19], "¿Cómo se llama la mesa de ayuda del laboratorio?", "Se llama \"MesaCóndor\". (Art. 19)", ["MesaCóndor"]),
    ("val", "simple", [21], "¿Cuál es la sanción por una falta grave?", "Suspensión de 30 días. (Art. 21)", ["30 días"]),
    ("val", "simple", [22], "¿Instalar software no autorizado es falta leve?", "No, instalar software no autorizado es una falta grave. (Art. 22)", ["grave"]),
    ("val", "simple", [23], "¿Cuál es el plazo para apelar?", "Tienes 5 días hábiles desde la notificación de la sanción. (Art. 23)", ["5 días hábiles"]),
    ("val", "simple", [25], "¿En qué fecha entró en vigencia el reglamento?", "El 1 de febrero de 2026. (Art. 25)", ["1 de febrero de 2026"]),
    ("val", "combinada", [7, 11], "Quiero reservar 30 horas seguidas de GPU, ¿puedo?", "Una reserva normal dura máximo 8 horas; para más tiempo necesitas una \"Corrida Extendida\" (formulario LCA-07), de hasta 72 horas. (Arts. 7 y 11)", ["8 horas", "LCA-07", "72 horas"]),
    ("val", "combinada", [21, 22], "¿Qué sanción corresponde a compartir credenciales?", "Compartir credenciales es una falta grave y se sanciona con suspensión de 30 días. (Arts. 21 y 22)", ["grave", "30 días"]),
    ("val", "fuera_alcance", [], "¿Cuál es el horario de la biblioteca?", NO, ["no regula"]),
    ("val", "fuera_alcance", [], "¿Dónde compro la camiseta oficial de la facultad?", NO, ["no regula"]),
    ("val", "fuera_alcance", [], "¿Cuántos créditos tiene la materia de minería multimedia?", NO, ["no regula"]),

    # ---------------- TEST (20) ----------------
    ("test", "simple", [2], "¿Un profesor puede usar el laboratorio?", "Sí, los docentes son usuarios autorizados del LCA. (Art. 2)", ["docentes"]),
    ("test", "simple", [5], "¿Cada cuántos días vence mi contraseña del LCA?", "La contraseña debe cambiarse cada 90 días. (Art. 5)", ["90 días"]),
    ("test", "simple", [7], "¿En qué plataforma se agendan las GPUs y con cuánto tiempo de antelación?", "En el portal \"Agenda Cóndor\", con al menos 48 horas de anticipación. (Art. 7)", ["Agenda Cóndor", "48 horas"]),
    ("test", "simple", [8], "Soy estudiante de maestría, ¿cuántas GPUs puedo usar simultáneamente?", "Como estudiante de posgrado puedes usar como máximo 4 GPUs simultáneas. (Art. 8)", ["4 GPUs"]),
    ("test", "simple", [9], "¿Cuántas horas-GPU semanales tiene un estudiante de doctorado?", "Posgrado tiene una cuota de 60 horas-GPU por semana. (Art. 9)", ["60 horas-GPU"]),
    ("test", "simple", [10], "Tengo una reserva en 3 horas y no podré ir, ¿estoy a tiempo de cancelarla?", "No, las reservas deben cancelarse con al menos 6 horas de anticipación. (Art. 10)", ["6 horas"]),
    ("test", "simple", [11], "¿Quién autoriza los trabajos de más de 8 horas y con qué formulario?", "El coordinador del laboratorio, mediante el formulario LCA-07 de \"Corrida Extendida\". (Art. 11)", ["coordinador", "LCA-07"]),
    ("test", "simple", [12], "Mi sustentación de tesis es en dos semanas y alguien de un curso quiere la misma GPU, ¿quién tiene prioridad?", "Tú: las tesis con sustentación en menos de 30 días tienen la máxima prioridad, por encima de los cursos. (Art. 12)", ["tesis", "30 días"]),
    ("test", "simple", [14], "¿A qué hora se hace el respaldo diario de /home?", "A las 2:00, todos los días. (Art. 14)", ["2:00"]),
    ("test", "simple", [16], "¿Qué herramientas puedo usar para instalar paquetes?", "Solo entornos de usuario: uv, conda o contenedores Apptainer; Docker no está permitido. (Art. 16)", ["uv", "conda", "Apptainer"]),
    ("test", "simple", [18], "¿Puedo entrar con un café sin tapa a la sala de estaciones?", "No, en la sala de estaciones solo se permiten bebidas en recipientes con tapa. (Art. 18)", ["con tapa"]),
    ("test", "simple", [20], "¿Qué día del mes hay mantenimiento programado?", "El primer miércoles de cada mes, de 6:00 a 12:00. (Art. 20)", ["primer miércoles"]),
    ("test", "simple", [22], "¿Minar criptomonedas qué tipo de falta es?", "Es una falta gravísima. (Art. 22)", ["gravísima"]),
    ("test", "simple", [24], "Publiqué un artículo usando el LCA, ¿qué debo hacer?", "Incluir el agradecimiento \"Este trabajo usó el Nodo Cóndor (LCA)\" y registrarlo ante la coordinación dentro de los 30 días siguientes. (Art. 24)", ["Este trabajo usó el Nodo Cóndor", "30 días"]),
    ("test", "combinada", [8, 9], "¿Cuántas GPUs a la vez y cuántas horas-GPU por semana tiene pregrado?", "Pregrado puede usar como máximo 2 GPUs simultáneas y tiene 20 horas-GPU por semana. (Arts. 8 y 9)", ["2 GPUs", "20 horas-GPU"]),
    ("test", "combinada", [13, 14], "Si guardo mis resultados en /scratch, ¿están seguros?", "No: /scratch es temporal, se borra cada 14 días y no tiene respaldo; usa /home, que se respalda diariamente. (Arts. 13 y 14)", ["14 días", "no tiene respaldo"]),
    ("test", "combinada", [17, 21, 22], "¿Qué sanción recibe quien mina criptomonedas en el laboratorio?", "Minar criptomonedas está prohibido y es falta gravísima: suspensión de 6 meses y reporte a la facultad. (Arts. 17, 21 y 22)", ["gravísima", "6 meses"]),
    ("test", "fuera_alcance", [], "¿Dónde queda la enfermería del campus?", NO, ["no regula"]),
    ("test", "fuera_alcance", [], "¿Cómo pago la matrícula del semestre?", NO, ["no regula"]),
    ("test", "fuera_alcance", [], "¿El laboratorio ofrece clases de inglés?", NO, ["no regula"]),
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data"
    counters = {"train": 0, "val": 0, "test": 0}
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in counters}
    for split, tipo, arts, q, a, hechos in E:
        counters[split] += 1
        row = {
            "id": f"{split}-{counters[split]:03d}",
            "question": q,
            "answer": a,
            "articulos": arts,
            "tipo": tipo,
            "hechos_clave": hechos,
            "origen": "sintetico (reglamento ficticio)",
        }
        files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    print(counters)


if __name__ == "__main__":
    main()
