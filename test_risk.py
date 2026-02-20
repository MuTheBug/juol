import unittest
from risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(initial_equity=1000.0)

    def test_base_leverage(self):
        self.assertEqual(self.rm.get_base_leverage(0), 10)
        self.assertEqual(self.rm.get_base_leverage(1), 5)
        self.assertEqual(self.rm.get_base_leverage(2), 3)

    def test_kelly_size_default(self):
        # With no history, should return 0.05
        self.assertEqual(self.rm.calculate_kelly_size(), 0.05)

    def test_kelly_size_with_history(self):
        # 60% win rate, 2:1 reward/risk
        # win_rate=0.6, avg_win=0.02, loss_rate=0.4, avg_loss=0.01
        # Kelly = (0.6 * 0.02 - 0.4 * 0.01) / 0.02 = (0.012 - 0.004) / 0.02 = 0.008 / 0.02 = 0.4
        # Half Kelly = 0.2
        for _ in range(6): self.rm.add_trade_result(0.02)
        for _ in range(4): self.rm.add_trade_result(-0.01)

        self.assertAlmostEqual(self.rm.calculate_kelly_size(), 0.2)

    def test_confidence_scaling(self):
        params = self.rm.calculate_position_parameters(price=1.0, atr=0.02, confidence=0.7, regime=0)
        # confidence=0.7 -> scale = (0.7-0.5)*2 = 0.4
        # base_leverage (regime 0) = 10
        # leverage = 10 * 0.4 = 4.0
        self.assertEqual(params['leverage'], 4.0)

    def test_circuit_breaker_drawdown(self):
        self.rm.update_equity(-200) # Equity 800, peak 1000. DD = 20%
        params = self.rm.calculate_position_parameters(price=1.0, atr=0.02, confidence=1.0, regime=0)
        # Kelly should be halved. Original 0.05 -> 0.025
        self.assertEqual(params['kelly_fraction'], 0.025)

        self.rm.update_equity(-100) # Equity 700, peak 1000. DD = 30%
        self.assertTrue(self.rm.check_circuit_breakers())
        self.assertTrue(self.rm.is_halted)

if __name__ == '__main__':
    unittest.main()
