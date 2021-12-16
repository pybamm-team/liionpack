#
# Tests the logger class.
#
import liionpack as lp
import unittest
import os


class TestLogger(unittest.TestCase):
    def test_logger(self):
        logger = lp.logger
        self.assertEqual(logger.level, 30)
        lp.set_logging_level("INFO")
        self.assertEqual(logger.level, 20)
        lp.set_logging_level("ERROR")
        self.assertEqual(logger.level, 40)
        lp.set_logging_level("VERBOSE")
        self.assertEqual(logger.level, 15)
        lp.set_logging_level("NOTICE")
        self.assertEqual(logger.level, 25)
        lp.set_logging_level("SUCCESS")
        self.assertEqual(logger.level, 35)

        lp.set_logging_level("SPAM")
        self.assertEqual(logger.level, 5)
        lp.logger.spam("Test spam level")
        lp.logger.verbose("Test verbose level")
        lp.logger.notice("Test notice level")
        lp.logger.success("Test success level")

        # reset
        lp.set_logging_level("WARNING")

    def test_log_to_file(self):
        cwd = os.getcwd()
        filename = os.path.join(cwd, "temp_log_file.log")
        with open(filename, "w"):
            lp.log_to_file(filename)
            lp.logger.warning("This should write to file")
            assert os.path.isfile(filename)
        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
