from datetime import timedelta

from airalert import config


def test_splits_are_ordered_in_time():
    # AC-5.6: train → validation → test, без перетину
    assert config.TRAIN_START < config.VAL_START < config.TEST_START < config.TEST_END


def test_splits_leave_room_for_gap():
    # AC-5.6: між вибірками має вміститися проміжок у h годин
    gap = timedelta(hours=max(config.HORIZONS_H))
    assert config.VAL_START - config.TRAIN_START > gap
    assert config.TEST_START - config.VAL_START > gap


def test_test_period_ends_before_outages():
    # AC-1.3: тестовий період MVP не зачіпає днів збою джерела
    assert all(config.TEST_END <= start for start, _, _ in config.OUTAGES)
