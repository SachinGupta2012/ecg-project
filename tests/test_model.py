"""Tests for CNN baseline model."""
import torch


class TestCNNBaseline:
    """Tests for the CNN baseline model."""

    def test_model_forward_pass(self):
        """Model should produce correct output shape."""
        from src.models.cnn_baseline import CNNBaseline

        model = CNNBaseline(num_classes=5)
        model.eval()

        batch = torch.randn(4, 1, 288)
        with torch.no_grad():
            output = model(batch)

        assert output.shape == (4, 5)

    def test_model_parameter_count(self):
        """Model should have reasonable parameter count."""
        from src.models.cnn_baseline import CNNBaseline

        model = CNNBaseline(num_classes=5)
        params = sum(p.numel() for p in model.parameters())

        assert params > 100_000
        assert params < 5_000_000

    def test_model_probabilities_sum_to_one(self):
        """Softmax output should sum to 1."""
        from src.models.cnn_baseline import CNNBaseline

        model = CNNBaseline(num_classes=5)
        model.eval()

        batch = torch.randn(2, 1, 288)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

        assert probs.shape == (2, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)

    def test_single_beat_prediction(self):
        """Model should handle single beat input."""
        from src.models.cnn_baseline import CNNBaseline

        model = CNNBaseline(num_classes=5)
        model.eval()

        beat = torch.randn(1, 1, 288)
        with torch.no_grad():
            output = model(beat)

        assert output.shape == (1, 5)
        pred_class = output.argmax(dim=1).item()
        assert 0 <= pred_class < 5
