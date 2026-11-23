import unittest

from edge_sim.protection import FaultProtection


class FaultProtectionTests(unittest.TestCase):
    def test_requires_two_matching_high_confidence_fault_windows(self):
        protection = FaultProtection()
        self.assertIsNone(protection.update(1, 0.90, "Bearing Wear"))
        self.assertIsNone(protection.update(2, 0.95, "Imbalance"))
        event = protection.update(2, 0.91, "Imbalance")
        self.assertEqual(event.event, "trip")
        self.assertTrue(event.relay_cutoff)
        self.assertTrue(event.buzzer)
        self.assertTrue(event.red_led)

    def test_low_confidence_resets_fault_streak(self):
        protection = FaultProtection()
        protection.update(1, 0.90, "Bearing Wear")
        self.assertIsNone(protection.update(1, 0.85, "Bearing Wear"))
        self.assertIsNone(protection.update(1, 0.91, "Bearing Wear"))
        self.assertFalse(protection.active)
        self.assertEqual(protection.update(1, 0.92, "Bearing Wear").event, "trip")

    def test_three_normal_windows_recover_and_reset_actuators(self):
        protection = FaultProtection()
        protection.update(1, 0.9, "Bearing Wear")
        protection.update(1, 0.9, "Bearing Wear")
        self.assertIsNone(protection.update(0, 0.9, "Normal"))
        self.assertIsNone(protection.update(0, 0.9, "Normal"))
        event = protection.update(0, 0.9, "Normal")
        self.assertEqual(event.event, "recovery")
        self.assertFalse(event.relay_cutoff)
        self.assertFalse(protection.active)


if __name__ == "__main__":
    unittest.main()
