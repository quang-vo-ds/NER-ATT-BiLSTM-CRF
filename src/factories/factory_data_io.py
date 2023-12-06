
from src.data_io.data_io_cnbi import DataIONCBI

class DataIOFactory():
    """DataIOFactory contains wrappers to create various data readers/writers."""
    @staticmethod
    def create(args):
        if args.data_io == 'ncbi_disease':
            return DataIONCBI(dataset_name = args.data_io, 
                              train_no = args.train_no, 
                              dev_no = args.dev_no,
                              test_no = args.test_no
                             )
        else:
            raise ValueError('Unknown DataIO %s.' % args.data_io)