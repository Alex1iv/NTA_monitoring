from clickhouse_driver import Client, errors
import time

class DBWriter:
    """Подключение, импорт и экспорт данных в ClickHouse
    """    
    def __init__(self, config:dict, username:str, password:str, logger=None):
        """Initiate class

        Args:
            config (dict): подключение к бд, хост, порт, название бд
            username (str): логин 
            password (str): пароль
            logging (_type_, optional): объект логгер. Defaults to None.
        """        

        self.client = None
        retries, delay = 3, 2
        
        for attempt in range(retries):
            try:
                self.client = Client(
                    host = config["host"],
                    port = config["port"],
                    database=config["dbname"],
                    user = username,
                    password = password
                )
                
                self.logger = logger
                
                self.client.execute('SELECT 1')
                
                if self.logger:
                    self.logger.info(f"[DB] Connected successfully")
                
                break
                    
            except errors.NetworkError as e:
                if attempt == retries - 1:
                    
                    raise Exception(f"Failed to connect to ClickHouse after {retries} attempts: {str(e)}")
                print("[DB] ERROR connecting to database:")
                print(str(e))
                
                print(f"Attempt {attempt+1} failed. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
                
                
    def import_data(self, query:str):
        """Import data from ClickHouse

        Args:
            query (_type_): SQL query to be executed

        Returns:
            pd.DataFrame: Data from SQL table
        """
        if self.logger:
            self.logger.info('[DB]      Import started') 
               
        return self.client.query_dataframe(query)

    def export_data(
        self,
        df, #:pd.DataFrame,
        query:str='INSERT INTO intervals (dt, feature_name, ci_low, ci_high) VALUES' 
        ):
        """Export data into a ClickHouse table

        Args:
            df (pd.DataFrame): 
                Data to insert
            query (str, optional): 
                SQL table data. Defaults to 'INSERT INTO intervals (
                    dt, feature_name, ci_low, ci_high) VALUES'.

        Raises:
            Exception: Print msg if error
        """ 
        try:    
            self.client.execute('SELECT 1')
            
            if self.logger:
                self.logger.info("[DB]      connection was established")
        
        except Exception as e:
            if self.logger:
                self.logger.info(f"[DB]      ERROR inserting events {str(e)}")
            
            raise 
        
        # исключаем загрузку пустых строк
        if df is None or df.empty:

            if self.logger:
                self.logger.info("[DB]      No ip ranges to insert")
            return
        
        try:
            if self.logger:
                self.logger.info(f"[DB]     Export started")

            # convert to a list
            data = list(df.itertuples(index=False, name=None)) 
   
            self.client.execute(query, data)
            
            if self.logger:
                self.logger.info(f"[DB]      Inserted {len(df)} rows")

        except Exception as e:
            #print('[DB] ERROR inserting events')
            
            if self.logger:
                self.logger.info(f"[DB]      ERROR inserting events {str(e)}")
            
            raise 
    
