def norm(sig, ref, norm_type: str = "None"):
    match norm_type:
        case "None":
            return sig
        case "Subtract":
            return sig - ref
        case "Divide":
            return sig / ref
        case "Normalise":
            return 100.0 * (1.0 - sig / ref)
        case _:
            return sig
