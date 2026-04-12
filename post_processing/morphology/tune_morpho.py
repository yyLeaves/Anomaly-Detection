#!/usr/bin/env python3
"""
Direct Morphology Tuning - No Extract or Mask Steps
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_tuning import MorphologyValidationPipelineSimplified


def main():
    """Main entry point"""
    
    config_path = Path(__file__).parent.parent / "config" / "morpho_val.yaml"
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        print("[INFO] Please create config/morpho_val.yaml")
        sys.exit(1)
    
    try:
        # Initialize and run simplified pipeline
        pipeline = MorphologyValidationPipelineSimplified(config_path)
        pipeline.run()
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()