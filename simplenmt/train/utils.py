import logging

def get_logger():
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler('log.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)