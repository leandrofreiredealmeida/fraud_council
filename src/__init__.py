import os
import sys
import logging

# Define o format de logging com placeholders para timestamp, log level, nome do módulo e mensagem
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Define o diretório onde os arquivos de log serão guardados
log_dir = "logs"
# Cria o caminho completo para o arquivo de log
log_filepath = os.path.join(log_dir, "logging.log")
# Certifica que a pasta de log existe (cria se ela não existe)
os.makedirs(log_dir, exist_ok=True)

# Configurações do logging
logging.basicConfig(
    level = logging.INFO,  
    format = logging_str,  
    handlers = [  
        logging.FileHandler(log_filepath),  
        logging.StreamHandler(sys.stdout)  # Output logs para o console
    ]
)

# Cria a instância do logger com o nome customizado
logger = logging.getLogger("fraudcouncillogger")