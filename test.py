import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    handlers=[logging.StreamHandler()])
logging.info('Hello World')
print('Hello')