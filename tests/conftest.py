"""Shared pytest set-up for the RAINS tests."""
import matplotlib

# Draw figures off-screen so the tests run on machines without a display.
matplotlib.use("Agg")
