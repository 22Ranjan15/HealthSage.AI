import sys

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        """
        :param error_message: Standard error message string
        :param error_detail: The sys module to extract traceback info
        """
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message,error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        """
        Extracts script name, line number, and error details from sys.exc_info()
        """
        _, _, exc_tb = error_detail.exc_info()

        # If the exception is caught but exc_tb is None (rare edge case)
        if exc_tb is None:
            return f"Error: {error_message}"
        
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        formatted_message = (
            f"Error occurred in python script name [{file_name}] "
            f"line number [{line_number}] error message [{error_message}]"
        )
        return formatted_message
    
    def __str__(self):
        return self.error_message