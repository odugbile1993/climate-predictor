"""
Climate Action Predictor - SDG 13: Climate Action
Machine Learning Model to Predict CO2 Emissions and Recommend Reduction Strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ClimateActionPredictor:
    """
    A machine learning system to predict CO2 emissions and provide
    climate action recommendations for SDG 13
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.predictions = None
        
    def generate_sample_data(self, n_countries=100):
        """
        Generate synthetic climate and economic data for demonstration
        In real scenario, this would be replaced with actual datasets
        """
        np.random.seed(42)
        
        data = {
            'country': [f'Country_{i}' for i in range(n_countries)],
            'gdp_per_capita': np.random.normal(15000, 8000, n_countries),
            'population': np.random.randint(1e6, 100e6, n_countries),
            'renewable_energy_percent': np.random.uniform(5, 60, n_countries),
            'industrial_output': np.random.normal(30, 15, n_countries),
            'forest_area_percent': np.random.uniform(10, 70, n_countries),
            'vehicle_per_1000': np.random.normal(300, 150, n_countries),
            'energy_consumption_per_capita': np.random.normal(3000, 1500, n_countries)
        }
        
        # Generate CO2 emissions based on features with some noise
        co2_emissions = (
            data['gdp_per_capita'] * 0.3 +
            data['industrial_output'] * 2.5 +
            data['vehicle_per_1000'] * 0.8 -
            data['renewable_energy_percent'] * 1.2 -
            data['forest_area_percent'] * 0.5 +
            np.random.normal(0, 50, n_countries)
        )
        
        data['co2_emissions_per_capita'] = np.maximum(co2_emissions, 0)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Preprocess the dataset for machine learning"""
        # Select features and target
        features = [
            'gdp_per_capita', 'population', 'renewable_energy_percent',
            'industrial_output', 'forest_area_percent', 'vehicle_per_1000',
            'energy_consumption_per_capita'
        ]
        
        X = df[features]
        y = df['co2_emissions_per_capita']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, features
    
    def train_models(self, X, y):
        """Train multiple ML models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_score = -np.inf
        best_model = None
        best_name = None
        
        print("Training Multiple Models...")
        print("=" * 50)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            self.models[name] = {
                'model': model,
                'r2_score': r2,
                'predictions': y_pred
            }
            
            print(f"{name}:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f}")
            print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            print("-" * 30)
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
        
        print(f"\n🎯 Best Model: {best_name} (R²: {best_score:.4f})")
        return best_model, best_name
    
    def analyze_feature_importance(self, model, feature_names):
        """Analyze and visualize feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # For linear models, use absolute coefficients
            importance = np.abs(model.coef_)
            
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        return self.feature_importance
    
    def predict_emissions_reduction(self, country_data, reduction_strategies):
        """
        Predict CO2 emissions reduction based on different strategies
        """
        original_emission = self.models['Random Forest']['model'].predict(
            self.scaler.transform([country_data])
        )[0]
        
        results = []
        for strategy, changes in reduction_strategies.items():
            modified_data = country_data.copy()
            for feature, change_percent in changes.items():
                feature_idx = self.feature_names.index(feature)
                modified_data[feature_idx] *= (1 + change_percent / 100)
            
            new_emission = self.models['Random Forest']['model'].predict(
                self.scaler.transform([modified_data])
            )[0]
            
            reduction = original_emission - new_emission
            reduction_percent = (reduction / original_emission) * 100
            
            results.append({
                'strategy': strategy,
                'original_emission': original_emission,
                'new_emission': new_emission,
                'reduction': reduction,
                'reduction_percent': reduction_percent
            })
        
        return results
    
    def visualize_results(self, X_test, y_test, predictions, feature_importance):
        """Create visualizations for model performance and insights"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Predictions vs Actual
        axes[0, 0].scatter(y_test, predictions, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual CO2 Emissions')
        axes[0, 0].set_ylabel('Predicted CO2 Emissions')
        axes[0, 0].set_title('Predictions vs Actual Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature Importance
        axes[0, 1].barh(feature_importance['feature'], feature_importance['importance'])
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Feature Importance for CO2 Emissions Prediction')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        residuals = y_test - predictions
        axes[1, 0].scatter(predictions, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residual Analysis')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Emission Distribution
        axes[1, 1].hist([y_test, predictions], bins=20, alpha=0.7, label=['Actual', 'Predicted'])
        axes[1, 1].set_xlabel('CO2 Emissions per Capita')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Actual vs Predicted Emissions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('assets/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete climate action analysis"""
        print("🌍 Climate Action Predictor - SDG 13 Analysis")
        print("=" * 60)
        
        # Generate and preprocess data
        print("1. 📊 Generating and preprocessing climate data...")
        df = self.generate_sample_data()
        X, y, self.feature_names = self.preprocess_data(df)
        
        # Train models
        print("2. 🤖 Training machine learning models...")
        best_model, best_name = self.train_models(X, y)
        
        # Feature importance
        print("3. 🔍 Analyzing feature importance...")
        feature_importance = self.analyze_feature_importance(best_model, self.feature_names)
        
        # Get test predictions for visualization
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        predictions = best_model.predict(X_test)
        
        # Visualize results
        print("4. 📈 Generating visualizations...")
        self.visualize_results(X_test, y_test, predictions, feature_importance)
        
        # Demonstrate prediction for reduction strategies
        print("5. 💡 Demonstrating emission reduction strategies...")
        sample_country = X[0]  # First country in dataset
        reduction_strategies = {
            'Renewable Energy Boost': {'renewable_energy_percent': 20},
            'Industrial Efficiency': {'industrial_output': -10, 'energy_consumption_per_capita': -15},
            'Sustainable Transport': {'vehicle_per_1000': -15, 'renewable_energy_percent': 10},
            'Forest Conservation': {'forest_area_percent': 15, 'industrial_output': -5}
        }
        
        reduction_results = self.predict_emissions_reduction(sample_country, reduction_strategies)
        
        print("\n" + "=" * 60)
        print("📋 EMISSION REDUCTION STRATEGIES ANALYSIS")
        print("=" * 60)
        for result in reduction_results:
            print(f"\n🌱 Strategy: {result['strategy']}")
            print(f"   Original Emission: {result['original_emission']:.1f} tons/capita")
            print(f"   New Emission: {result['new_emission']:.1f} tons/capita")
            print(f"   Reduction: {result['reduction']:.1f} tons/capita ({result['reduction_percent']:.1f}%)")
        
        return {
            'model': best_model,
            'feature_importance': feature_importance,
            'reduction_strategies': reduction_results,
            'performance': self.models[best_name]
        }

# Main execution
if __name__ == "__main__":
    # Initialize and run the climate predictor
    climate_predictor = ClimateActionPredictor()
    results = climate_predictor.run_complete_analysis()
    
    print("\n" + "=" * 60)
    print("✅ CLIMATE ACTION PREDICTOR ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\n💡 Key Insights for SDG 13:")
    print("• Machine learning can accurately predict CO2 emissions based on economic and environmental factors")
    print("• Feature importance analysis reveals key drivers of emissions")
    print("• The model can simulate the impact of various climate action strategies")
    print("• This approach supports evidence-based policy making for climate action")
