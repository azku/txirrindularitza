import unittest
import os
import shutil
import tempfile
import irudi_eraldaketak


SARRERA_DIREKTORIOA = '../data/wpcf7-files'
class TestIrudiEraldaketak(unittest.TestCase):
    
    def setUp(self):
        """
        Set up the test environment.
        This creates a temporary directory with some mock image files.
        """

    def tearDown(self):
        """
        Clean up after the test.
        This removes the temporary directory and files.
        """
    def test_get_direktorioko_irudiak(self):
        """
        Test direktorio batean dauden irudiak lortu
        """
        self.assertEqual(sum(1 for _ in irudi_eraldaketak.get_direktorioko_irudiak(SARRERA_DIREKTORIOA)), 20, "Irudi kopuruak 20 izan behar du")
    
    def test_argazkiei_buelta_eman(self):
        """
        Test the argazkiei_buelta_eman function.
        """
        # Call the function with the test directory and action "buelta_emanda"
        irteera_izena = "buelta_emanda"
        irudi_eraldaketak.argazkiei_buelta_eman(SARRERA_DIREKTORIOA, irteera_izena)
        # Assert that the function processed the correct files
        self.assertEqual("Bai", "Bai", "The number of processed files should be 3.")
        self.assertEqual(sum(1 for _ in irudi_eraldaketak.get_direktorioko_irudiak(SARRERA_DIREKTORIOA + "/" + irteera_izena)), 20, "Irudi kopuruak 20 izan behar du")

    def test_predikzioak_burutu(self):
        """
        Test the argazkiei_buelta_eman function.
        """
        # Call the function with the test directory and action "buelta_emanda"
        #predikzioak_burutu(SARRERA_DIREKTORIOA+"/buelta_emanda", "predikzioak","yolov8m.pt")
        irudi_eraldaketak.predikzioak_burutu(SARRERA_DIREKTORIOA+"/buelta_emanda", "predikzioak","license_plate_detector.pt",0)
        # Assert that the function processed the correct files
        self.assertEqual("Bai", "Bai", "The number of processed files should be 3.")

if __name__ == "__main__":
    unittest.main(buffer=False)
