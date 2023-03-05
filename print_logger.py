from datetime import datetime
def get_time():
    now_time = datetime.now()
    return str(now_time.year)+ '-' +str(now_time.month)+ '-' +str(now_time.day)+ '-' +str(now_time.hour)+ '-' +str(now_time.minute) + '-' +str(now_time.second)
import logging
logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                    filename = '{}.log'.format(get_time()),
                    filemode = 'a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    # a是追加模式，默认如果不写的话，就是追加模式
                    format = '%(asctime)s - %(message)s' # 日志格式
                    )
def test_log():
    for i in range(10):
        logging.info('hello!{}'.format(i))
test_log()
#
# import logging
#
# logging.debug('debug log test')
# logging.info('info log test')
# logging.warning('warning log test')
# logging.error('error log test')
# logging.critical('critical log test')