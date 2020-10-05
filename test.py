import unittest
import sys

if __name__ == '__main__':
    testsuite = unittest.TestLoader().discover('./test')
    s = unittest.TextTestRunner(verbosity=1).run(testsuite)
    flag = bool(s.failures) or bool(s.errors)
    sys.exit(flag)
