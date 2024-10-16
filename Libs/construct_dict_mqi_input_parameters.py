import random




def construct_dict_mqi_input_parameters(parent_dir, dicom_dir, ct_name = None, output_dir = None, GPUID = 0, particles_per_history = 10000):
    
    input_parameters = {"GPUID": GPUID,
                        "RandomSeed": int(random.random()*1e6),
#                        "RandomSeed": -1932780356, # (integer, use negative value if use current time)
                        "UseAbsolutePath": False,
                        "TotalThreads": -1, #(Integer, use negative value for using optimized number of threads)
                        "MaxHistoriesPerBatch": 0,
                        "Verbosity": 0,
                        ## Data directories
                        "ParentDir": parent_dir, # parent directory of DICOM files
                        "DicomDir": dicom_dir, # directory of the DICOM
                        ## Quantity to score
                        "Scorer": "Dose", # Dose is dose to water. We support dose to medium, dose to water, track-weighted LET, dose-weighted LET. Modification in environment is needed to add selection
                        "SupressStd": True,
                        "ReadStructure": False,
                        "ROIName": "External",
                        ## Source parameters
                        "SourceType": "FluenceMap",
                        "SimulationType": "perBeam",
                        "BeamNumbers": 0,
                        "ParticlesPerHistory": particles_per_history, # Scale factor to select the number of primary particles
                        "ScoreToCTGrid": True,
                        # "UnitWeights": 1000,
                        ## Output path
                        "OutputDir": output_dir, # output directory
                        "OutputFormat": "mhd", # output format (raw, mhd, mha, npz)
                        "OverwriteResults": True,
                        "RBE": 1.1,
                        "StoppingStatistics": False,
                        "RangeshifterDensity": 1.217,
                        }
    
    if ct_name:
        input_parameters["CTVolumeName"] = ct_name
    return input_parameters




