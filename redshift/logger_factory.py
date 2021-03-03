import logging


class LoggerFactory:
    @staticmethod
    def get_logger(name,
                   level=logging.INFO,
                   add_formatter=True,
                   add_file_handler=False,
                   **kwargs):
        """Return a generic logger"""
        _logger = logging.getLogger(name)
        _logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        formatter = None
        if add_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)
            sh.setLevel(level)

        if add_file_handler:
            filename = kwargs.get('filename', '{}-log.txt'.format(name))
            fh = logging.FileHandler(filename, mode='a+')
            fh.setLevel(kwargs.get('file_log_level', logging.DEBUG))
            if add_formatter:
                fh.setFormatter(formatter)
            _logger.addHandler(fh)
        _logger.addHandler(sh)
        return _logger