"""
Automated Verification Suite for JARVIS PyQt6 Background Service and HUD
"""
import sys
import unittest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel

# Import the implementation from jarvis_ui
import jarvis_ui
from jarvis_ui import JarvisHUD, AudioListenerThread


class TestJarvisHUD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.hud = JarvisHUD()

    def tearDown(self):
        self.hud.close()

    def test_window_flags_and_attributes(self):
        """Verify window flags: Frameless, AlwaysOnTop, Tool, and WA_TranslucentBackground."""
        flags = self.hud.windowFlags()
        self.assertTrue(flags & Qt.WindowType.FramelessWindowHint, "Missing FramelessWindowHint")
        self.assertTrue(flags & Qt.WindowType.WindowStaysOnTopHint, "Missing WindowStaysOnTopHint")
        self.assertTrue(flags & Qt.WindowType.Tool, "Missing Tool flag (for taskbar hiding)")
        self.assertTrue(
            self.hud.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground),
            "WA_TranslucentBackground is not enabled",
        )

    def test_startup_hidden(self):
        """Verify HUD is completely hidden on startup."""
        self.assertFalse(self.hud.isVisible(), "HUD should be hidden on startup")

    def test_label_styling_and_central_widget(self):
        """Verify UI consists of a single styled QLabel with neon green text & rounded corners."""
        central = self.hud.centralWidget()
        self.assertIsInstance(central, QLabel, "Central widget must be a QLabel")
        self.assertIs(central, self.hud.label, "Central widget must be the styled label")
        
        stylesheet = self.hud.label.styleSheet()
        self.assertIn("#00FF66", stylesheet, "Neon green color (#00FF66) missing in stylesheet")
        self.assertIn("border-radius", stylesheet, "Rounded corners missing in stylesheet")
        self.assertIn("rgba(10, 16, 26", stylesheet, "Dark translucent background missing in stylesheet")

    def test_bottom_center_positioning(self):
        """Verify HUD positions dynamically at bottom-center of primary screen."""
        screen = self.app.primaryScreen()
        avail = screen.availableGeometry()

        self.hud.on_wake()
        self.hud.on_update_text("Testing bottom center positioning with sample text")

        geom = self.hud.geometry()
        # Verify horizontal center alignment within tolerance
        expected_x = avail.x() + (avail.width() - geom.width()) // 2
        self.assertAlmostEqual(geom.x(), expected_x, delta=2, msg="HUD is not centered horizontally")

        # Verify bottom positioning
        expected_y = avail.y() + avail.height() - geom.height() - 45
        self.assertAlmostEqual(geom.y(), expected_y, delta=2, msg="HUD is not positioned near screen bottom")

    def test_behavior_flow(self):
        """Verify wake_signal -> show, update_text_signal -> setText, sleep_signal -> 4s timer -> hide."""
        # 1. wake_signal
        self.hud.on_wake()
        self.assertTrue(self.hud.isVisible(), "HUD must be visible on wake")

        # 2. update_text_signal
        test_text = "⚡ JARVIS Activated"
        self.hud.on_update_text(test_text)
        self.assertIn("JARVIS Activated", self.hud.label.text())

        # 3. sleep_signal
        self.hud.on_sleep()
        self.assertTrue(self.hud.sleep_timer.isActive(), "Sleep timer should be active")
        self.assertEqual(self.hud.sleep_timer.interval(), 4000, "Sleep timer interval must be 4000ms (4 seconds)")

        # 4. Timer expiry triggers fade out and hide
        self.hud.sleep_timer.stop()
        self.hud._start_fade_out()
        # Simulate fade finished
        self.hud.setWindowOpacity(0.0)
        self.hud._on_fade_finished()
        self.assertFalse(self.hud.isVisible(), "HUD must hide after sleep timeout")

    def test_audio_listener_thread_signals(self):
        """Verify AudioListenerThread has the exact required custom signals."""
        thread = AudioListenerThread()
        self.assertTrue(hasattr(thread, "wake_signal"), "AudioListenerThread missing wake_signal")
        self.assertTrue(hasattr(thread, "update_text_signal"), "AudioListenerThread missing update_text_signal")
        self.assertTrue(hasattr(thread, "sleep_signal"), "AudioListenerThread missing sleep_signal")


if __name__ == "__main__":
    unittest.main()
