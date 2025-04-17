# This function defines the ODMR sequence for the pulseblaster


def seq_cw_lockin_esr(
    system,
    int_time: float = 50.0,
    n_cycles: int = 1,
    n_freq: int = 1,
):
    trigger_time = system.sequence_params["camera_trig_time"]  # s

    # convert times to ns assuming the user inputs in s
    int_time = int_time * 1e9
    trigger_time = trigger_time * 1e9

    # Start the programming of the pulseblaster
    pb = system.pb
    pb.start_programming()

    # Turn the laser on to initialize the system
    pb.add_pulse(["laser", "mw_x"], int_time)

    for i in range(0, n_cycles * n_freq):
        # for the camera lockin we take two frames of each state to
        # get a better fourier transform of the signal

        # signal f1
        # pb.add_pulse(['laser', 'mw_x'], int_time, const_trigger = ["camera"])
        # pb.add_pulse(['laser', 'mw_x'], trigger_time)

        # signal f1
        pb.add_pulse(["laser", "mw_x"], int_time, const_trigger=["camera"])
        pb.add_pulse(["laser", "mw_x"], trigger_time, const_trigger=["rf_trig"])

        # signal f2
        # pb.add_pulse(['laser', 'mw_x'], int_time, const_trigger = ["camera"])
        # pb.add_pulse(['laser', 'mw_x'], trigger_time)

        # signal f2
        pb.add_pulse(["laser", "mw_x"], int_time, const_trigger=["camera"])
        pb.add_pulse(["laser", "mw_x"], trigger_time, const_trigger=["rf_trig"])
    # Turn the laser off and end sequence
    pb.end_sequence(1e6)

    # End of pulse program
    system.pb.stop_programming()
