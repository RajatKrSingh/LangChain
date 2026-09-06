import os
from dotenv import load_dotenv 

def read_env_variables(CONFIG_PATH: [str] = "config/.env"):
    ''' Read Environment variables
    '''
    load_dotenv(dotenv_path=CONFIG_PATH)

def main():
    read_env_variables()

if __name__=="__main__":
    main()