import sys
import glob

class ExtractData():

    def __init__(self):
        self.path_to_data = ""
        self.video_dimensions = ""
        self.data_files = []

    def get_data_files(self):

        for filename in glob.glob('*.JSON'):
            self.data_files.append(filename)
        

def main(argv=None):
    data = ExtractData()


if __name__ == '__main__':
    main(sys.argv[1:])