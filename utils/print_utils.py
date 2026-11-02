def print_results(step, logger, config):
    if config.verbose:
        string = f"Step {int(step)}: "
        if "logZ/delta" in logger and len(logger["logZ/delta"]) > 0:
            string += f"logZ_delta {float(logger['logZ/delta'][-1]):.4f}; "
        if "Z/delta" in logger and len(logger["Z/delta"]) > 0:
            string += f"Z_delta {float(logger['Z/delta'][-1]):.4f}; "
        if "logZ/value" in logger and len(logger["logZ/value"]) > 0:
            string += f"logZ {float(logger['logZ/value'][-1]):.4f}; "
        if "Z/value" in logger and len(logger["Z/value"]) > 0:
            string += f"Z {float(logger['Z/value'][-1]):.4f}; "

        print(string)
