# Add this to helpers/simulation_logger.py

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from io import StringIO
import sys

from helpers import classification as pf

class SimulationLogger:
    def __init__(self, league, base_path="../dist/simulations"):
        self.league = league
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.simulation_dir = os.path.join(base_path, league, self.timestamp)
        
        # Create directory structure
        os.makedirs(self.simulation_dir, exist_ok=True)
        
        # Initialize storage
        self.config = {}
        self.outputs = []
        self.charts = []
        
        print(f"📁 Simulation directory created: {self.simulation_dir}")
    
    def log_config(self, **kwargs):
        """Log all configuration parameters"""
        self.config.update(kwargs)
        
        # Save config immediately
        config_path = os.path.join(self.simulation_dir, "simulation_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def capture_print(self, func, description=""):
        """Capture print outputs from a function"""
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            result = func()
            output = captured_output.getvalue()
            
            # Store output
            self.outputs.append({
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "output": output
            })
            
            return result
        finally:
            sys.stdout = old_stdout
        
            # Also print to console
            print(output, end='')
    
    def save_chart(self, chart_name, description="", figure=None):
        """Save a specific matplotlib figure"""
        
        if figure is None:
            # Get the current figure
            figure = plt.gcf()
        
        chart_path = os.path.join(self.simulation_dir, f"{chart_name}.png")
        figure.savefig(chart_path, dpi=300, bbox_inches='tight')
        
        self.charts.append({
            "name": chart_name,
            "description": description,
            "path": chart_path,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"Chart saved: {chart_name}.png")
        return chart_path
    
    def save_all_open_charts(self, base_name, description=""):
        """Save all currently open matplotlib figures"""
        
        chart_count = 0
        for i in plt.get_fignums():
            fig = plt.figure(i)
            chart_path = os.path.join(self.simulation_dir, f"{base_name}_{i}.png")
            fig.savefig(chart_path, dpi=300, bbox_inches='tight')
            
            self.charts.append({
                "name": f"{base_name}_{i}",
                "description": f"{description} - Figure {i}",
                "path": chart_path,
                "timestamp": datetime.now().isoformat()
            })
            chart_count += 1
        
        if chart_count == 0:
            print(f"No charts found to save for {base_name}")
        else:
            print(f"Saved {chart_count} charts: {base_name}_*.png")
        
        return chart_count
    
    def save_dataframe(self, df, filename, description=""):
        """Save DataFrame to CSV"""
        file_path = os.path.join(self.simulation_dir, f"{filename}.csv")
        df.to_csv(file_path, index=False)
        
        print(f"💾 DataFrame saved: {filename}.csv")

    def save_model_parameters(self, last_season_models, markets):
        """Save detailed model parameters for all markets"""
        model_params_path = os.path.join(self.simulation_dir, "model_parameters.json")
        
        model_details = {}
        
        for market_name in markets.keys():
            if market_name in last_season_models:
                model_details[market_name] = {}
                
                for model_name, model_info in last_season_models[market_name].items():
                    model_details[market_name][model_name] = {
                        "estimator_class": model_info["estimator"].__class__.__name__,
                        "parameters": model_info.get("params"),
                        "score": model_info.get("score"),
                        "pipeline_steps": list(model_info["pipeline"].named_steps.keys()) if "pipeline" in model_info else None
                    }
                    
                    # Get pipeline parameters if available
                    if "pipeline" in model_info:
                        try:
                            pipeline_params = {}
                            for step_name, step in model_info["pipeline"].named_steps.items():
                                pipeline_params[step_name] = step.get_params()
                            model_details[market_name][model_name]["pipeline_parameters"] = pipeline_params
                        except Exception as e:
                            model_details[market_name][model_name]["pipeline_parameters"] = f"Error extracting: {str(e)}"
        
        # Save to JSON
        with open(model_params_path, 'w') as f:
            json.dump(model_details, f, indent=2, default=str)
        
        print(f"🔧 Model parameters saved: model_parameters.json")
        return model_details
    
    def save_model_accuracies(self, matches, models, markets):
        """Save model accuracy outputs"""
        accuracy_path = os.path.join(self.simulation_dir, "model_accuracies.txt")
        
        with open(accuracy_path, 'w') as f:
            for market_name in markets:
                f.write(f"\n{market_name.upper()} Model Accuracies:\n")
                f.write("="*50 + "\n")
                
                # Capture the accuracy function output
                old_stdout = sys.stdout
                sys.stdout = f
                pf.show_classification_accuracies(matches, models, result_col=market_name)
                sys.stdout = old_stdout
    
    def finalize(self, simulation_summary=None):
        """Save final summary and close logger"""
        # Save all captured outputs
        outputs_path = os.path.join(self.simulation_dir, "captured_outputs.txt")
        with open(outputs_path, 'w') as f:
            for output in self.outputs:
                f.write(f"\n{'='*60}\n")
                f.write(f"Description: {output['description']}\n")
                f.write(f"Timestamp: {output['timestamp']}\n")
                f.write(f"{'='*60}\n")
                f.write(output['output'])
                f.write("\n")
        
        # Save simulation summary
        if simulation_summary:
            summary_path = os.path.join(self.simulation_dir, "simulation_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(simulation_summary, f, indent=2, default=str)
        
        # Save chart metadata
        charts_path = os.path.join(self.simulation_dir, "charts_metadata.json")
        with open(charts_path, 'w') as f:
            json.dump(self.charts, f, indent=2)
        
        print(f"✅ Simulation logged to: {self.simulation_dir}")
        return self.simulation_dir