# FairLens Insight: Model Fairness Analysis Tool
# This implementation analyzes fairness in a classification model for income prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import shap
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.metrics import selection_rate, MetricFrame
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

class FairLensInsight:
    """
    A tool to analyze fairness in classification models, with particular focus on
    detecting and visualizing bias against protected attributes.
    """
    
    def __init__(self, model_type="gradient_boosting"):
        """
        Initialize the FairLens tool.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use. Options: "gradient_boosting", "logistic_regression"
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_features = None
        self.feature_names = None
        self.results = {}
    
    def load_adult_census_data(self):
        """
        Load and preprocess the UCI Adult Census Income dataset.
        """
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]
        
        print("Loading UCI Adult Census Income dataset...")
        
        # Try to load from scikit-learn datasets
        try:
            from sklearn.datasets import fetch_openml
            adult = fetch_openml(name='adult', version=1, as_frame=True)
            df = adult.data
            df['income'] = adult.target
        except:
            # Fallback to loading from UCI repository
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            df = pd.read_csv(url, names=column_names, sep=', ', engine='python')
        
        # Clean the data
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convert target to binary
        df['income'] = df['income'].map({'>50K': 1, '<=50K': 0, '>50K.': 1, '<=50K.': 0})
        
        # Identify sensitive attributes
        self.sensitive_features = {
            'sex': df['sex'],
            'race': df['race']
        }
        
        # Features and target
        X = df.drop('income', axis=1)
        y = df['income']
        
        # Keep track of original feature names
        self.feature_names = X.columns.tolist()
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Also create test sets with sensitive attributes for evaluation
        self.sensitive_test = {
            'sex': self.sensitive_features['sex'].loc[self.X_test.index],
            'race': self.sensitive_features['race'].loc[self.X_test.index]
        }
        
        print(f"Data loaded successfully. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        # Show dataset summary
        self._display_dataset_summary(df)
        
        return df
    
    def _display_dataset_summary(self, df):
        """Display summary information about the dataset."""
        print("\n--- Dataset Summary ---")
        print(f"Total records: {len(df)}")
        
        # Distribution of the target variable
        income_counts = df['income'].value_counts(normalize=True) * 100
        print("\nTarget distribution:")
        for income, percentage in income_counts.items():
            label = ">$50K" if income == 1 else "≤$50K"
            print(f"  {label}: {percentage:.1f}%")
        
        # Distribution of sensitive attributes
        print("\nSensitive attribute distributions:")
        
        print("\nGender distribution:")
        gender_counts = df['sex'].value_counts(normalize=True) * 100
        for gender, percentage in gender_counts.items():
            print(f"  {gender}: {percentage:.1f}%")
        
        print("\nRace distribution:")
        race_counts = df['race'].value_counts(normalize=True) * 100
        for race, percentage in race_counts.items():
            print(f"  {race}: {percentage:.1f}%")
        
        # Cross-tabulation of income by gender
        print("\nIncome distribution by gender:")
        gender_income = pd.crosstab(df['sex'], df['income'], normalize='index') * 100
        gender_income.columns = ["≤$50K", ">$50K"]
        print(gender_income)
        
        # Cross-tabulation of income by race
        print("\nIncome distribution by race:")
        race_income = pd.crosstab(df['race'], df['income'], normalize='index') * 100
        race_income.columns = ["≤$50K", ">$50K"]
        print(race_income)
    
    def _create_preprocessor(self):
        """Create a column transformer for preprocessing the data."""
        # Identify categorical and numerical columns
        categorical_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing steps
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Get feature names after transformation
        self.preprocessor.fit(self.X_train)
        
        # Extract feature names after one-hot encoding
        encoded_categorical_cols = []
        if categorical_cols:
            encoder = self.preprocessor.named_transformers_['cat']
            categories = encoder.categories_
            for i, category in enumerate(categories):
                col_name = categorical_cols[i]
                for cat_value in category:
                    encoded_categorical_cols.append(f"{col_name}_{cat_value}")
        
        self.encoded_feature_names = numerical_cols + encoded_categorical_cols
    
    def train_model(self):
        """Train the classification model with preprocessing."""
        self._create_preprocessor()
        
        # Select model based on configuration
        if self.model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create and train the pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        print(f"\nTraining {self.model_type.replace('_', ' ').title()} model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model trained successfully. Test accuracy: {accuracy:.4f}")
    
    def evaluate_fairness(self):
        """Evaluate fairness metrics across sensitive attributes."""
        y_pred = self.model.predict(self.X_test)
        probas = self.model.predict_proba(self.X_test)[:, 1]
        
        print("\n--- Fairness Evaluation ---")
        
        fairness_results = {}
        
        for attribute, values in self.sensitive_test.items():
            print(f"\nFairness metrics for {attribute}:")
            
            # Calculate selection rates by group
            selection_rates = {}
            unique_groups = values.unique()
            
            for group in unique_groups:
                group_mask = (values == group)
                group_selection_rate = selection_rate(y_true=self.y_test[group_mask], 
                                                     y_pred=y_pred[group_mask])
                selection_rates[group] = group_selection_rate
                print(f"  Selection rate for {group}: {group_selection_rate:.4f}")
            
            # Calculate demographic parity (statistical parity difference)
            dpd = demographic_parity_difference(
                y_true=self.y_test,
                y_pred=y_pred,
                sensitive_features=values
            )
            print(f"  Demographic Parity Difference: {dpd:.4f}")
            
            # Calculate equalized odds difference
            eod = equalized_odds_difference(
                y_true=self.y_test,
                y_pred=y_pred,
                sensitive_features=values
            )
            print(f"  Equalized Odds Difference: {eod:.4f}")
            
            # Group-wise confusion matrices and metrics
            metrics_by_group = {}
            for group in unique_groups:
                group_mask = (values == group)
                
                # Calculate metrics
                group_accuracy = accuracy_score(self.y_test[group_mask], y_pred[group_mask])
                group_cm = confusion_matrix(self.y_test[group_mask], y_pred[group_mask])
                
                if len(group_cm) > 1:  # Check if we have both classes in the group
                    tn, fp, fn, tp = group_cm.ravel()
                    group_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    group_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    group_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    group_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                else:
                    # Handle the case where one class is missing
                    if self.y_test[group_mask].mean() == 0:  # All negative
                        tn = group_cm[0, 0]
                        fp, fn, tp = 0, 0, 0
                        group_fpr, group_fnr = 0, 0
                        group_precision, group_recall = 0, 0
                    else:  # All positive
                        tp = group_cm[0, 0]
                        tn, fp, fn = 0, 0, 0
                        group_fpr, group_fnr = 0, 0
                        group_precision, group_recall = 1, 1
                
                metrics_by_group[group] = {
                    'accuracy': group_accuracy,
                    'precision': group_precision,
                    'recall': group_recall,
                    'false_positive_rate': group_fpr,
                    'false_negative_rate': group_fnr,
                    'confusion_matrix': group_cm
                }
                
                print(f"\n  Metrics for {group}:")
                print(f"    Accuracy: {group_accuracy:.4f}")
                print(f"    Precision: {group_precision:.4f}")
                print(f"    Recall: {group_recall:.4f}")
                print(f"    False Positive Rate: {group_fpr:.4f}")
                print(f"    False Negative Rate: {group_fnr:.4f}")
            
            # Store results
            fairness_results[attribute] = {
                'selection_rates': selection_rates,
                'demographic_parity_difference': dpd,
                'equalized_odds_difference': eod,
                'metrics_by_group': metrics_by_group
            }
        
        self.results['fairness'] = fairness_results
        
        # Visualize fairness metrics
        self._visualize_fairness_metrics(fairness_results)
    
    def _visualize_fairness_metrics(self, fairness_results):
        """Visualize fairness metrics for different groups."""
        # 1. Selection rate comparison
        plt.figure(figsize=(12, 6))
        
        for i, (attribute, results) in enumerate(fairness_results.items()):
            plt.subplot(1, 2, i+1)
            groups = list(results['selection_rates'].keys())
            rates = list(results['selection_rates'].values())
            
            bars = plt.bar(groups, rates, color=sns.color_palette("viridis", len(groups)))
            plt.axhline(y=sum(rates)/len(rates), color='red', linestyle='--', 
                        label='Average selection rate')
            
            plt.ylim(0, max(rates) * 1.2)
            plt.title(f'Selection Rates by {attribute.title()}')
            plt.ylabel('Selection Rate')
            plt.legend()
            
            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('selection_rates_by_group.png')
        plt.close()
        
        # 2. Metric comparison across groups
        for attribute, results in fairness_results.items():
            metrics_to_plot = ['accuracy', 'precision', 'recall', 
                            'false_positive_rate', 'false_negative_rate']
            
            plt.figure(figsize=(15, 10))
            plt.suptitle(f'Performance Metrics by {attribute.title()}', fontsize=16)
            
            for i, metric in enumerate(metrics_to_plot):
                plt.subplot(2, 3, i+1)
                
                groups = []
                values = []
                
                for group, metrics in results['metrics_by_group'].items():
                    groups.append(group)
                    values.append(metrics[metric])
                
                bars = plt.bar(groups, values, color=sns.color_palette("viridis", len(groups)))
                plt.ylim(0, 1.0)
                plt.title(f'{metric.replace("_", " ").title()}')
                
                # Add values on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f'fairness_metrics_{attribute}.png')
            plt.close()
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance for different groups."""
        print("\n--- Feature Importance Analysis ---")
        
        # Get preprocessed data
        X_test_preprocessed = self.preprocessor.transform(self.X_test)
        
        # Initialize SHAP explainer
        print("Calculating SHAP values...")
        
        # Get the actual classifier from the pipeline
        classifier = self.model.named_steps['classifier']
        
        # Different approach based on model type
        if self.model_type == "gradient_boosting":
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_preprocessed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For multi-class, take the positive class
        else:  # Logistic regression or other
            explainer = shap.LinearExplainer(classifier, X_test_preprocessed)
            shap_values = explainer.shap_values(X_test_preprocessed)
        
        # Create feature names after preprocessing
        feature_names = []
        
        # For numerical features
        num_transformer = self.preprocessor.named_transformers_['num']
        num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_names.extend(num_cols)
        
        # For categorical features
        cat_transformer = self.preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_feature_names = cat_transformer.get_feature_names_out()
            feature_names.extend(cat_feature_names)
        elif hasattr(cat_transformer, 'categories_'):
            cat_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            for i, categories in enumerate(cat_transformer.categories_):
                for category in categories:
                    feature_names.append(f"{cat_cols[i]}_{category}")
        
        # Analyze SHAP values by sensitive attribute
        self.results['shap'] = {}
        
        for attribute, values in self.sensitive_test.items():
            print(f"\nAnalyzing feature importance for {attribute}...")
            
            # Get unique values of the sensitive attribute
            unique_groups = values.unique()
            
            # Store group-specific SHAP values
            group_shap_values = {}
            for group in unique_groups:
                group_mask = (values == group).values
                if sum(group_mask) > 0:  # Check if we have samples for this group
                    group_shap_values[group] = shap_values[group_mask]
            
            self.results['shap'][attribute] = group_shap_values
            
            # Visualize SHAP values for each group
            self._visualize_shap_by_group(attribute, group_shap_values, feature_names)
    
    def _visualize_shap_by_group(self, attribute, group_shap_values, feature_names):
        """Visualize SHAP values for different groups."""
        
        # 1. SHAP summary plots for each group
        plt.figure(figsize=(10 * len(group_shap_values), 8))
        
        for i, (group, values) in enumerate(group_shap_values.items()):
            plt.subplot(1, len(group_shap_values), i+1)
            
            # Create SHAP summary plot
            shap.summary_plot(values, features=feature_names, feature_names=feature_names, 
                             plot_type="bar", show=False, plot_size=(8, 8))
            plt.title(f'Feature Importance for {attribute.title()} = {group}')
            
        plt.tight_layout()
        plt.savefig(f'shap_importance_by_{attribute}.png')
        plt.close()
        
        # 2. Compare top features across groups
        top_n = 10  # Number of top features to compare
        
        # Calculate mean absolute SHAP values per feature for each group
        mean_shap_by_group = {}
        all_top_features = set()
        
        for group, values in group_shap_values.items():
            mean_abs_shap = np.abs(values).mean(axis=0)
            feature_importance = dict(zip(feature_names, mean_abs_shap))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            mean_shap_by_group[group] = feature_importance
            all_top_features.update([f for f, _ in top_features])
        
        # Create a comparison DataFrame
        comparison_data = []
        for feature in all_top_features:
            row = {'Feature': feature}
            for group in group_shap_values.keys():
                row[group] = mean_shap_by_group[group].get(feature, 0)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(by=list(group_shap_values.keys()), 
                                                 key=lambda x: x.sum(), ascending=False)
        
        # Visualize feature importance comparison
        plt.figure(figsize=(14, 10))
        sns.heatmap(comparison_df.set_index('Feature')[group_shap_values.keys()], 
                   annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'Feature Importance Comparison by {attribute.title()}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_comparison_{attribute}.png')
        plt.close()
        
        # 3. Create bar plot for top features across groups
        plt.figure(figsize=(14, 8))
        
        # Get top N features across all groups
        top_features_df = comparison_df.head(top_n)
        
        # Plot grouped bar chart
        bar_width = 0.8 / len(group_shap_values)
        x = np.arange(len(top_features_df))
        
        for i, (group, color) in enumerate(zip(group_shap_values.keys(), 
                                             sns.color_palette("viridis", len(group_shap_values)))):
            plt.bar(x + i * bar_width, top_features_df[group], 
                   width=bar_width, label=group, color=color)
        
        plt.xlabel('Feature')
        plt.ylabel('Mean |SHAP Value|')
        plt.title(f'Top {top_n} Features Importance by {attribute.title()}')
        plt.xticks(x + bar_width * (len(group_shap_values) - 1) / 2, top_features_df['Feature'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'top_features_by_{attribute}.png')
        plt.close()
    
    def create_fairness_report(self):
        """Generate a comprehensive fairness report."""
        print("\n--- Fairness Report ---")
        
        # Summarize overall findings
        print("Summary of fairness analysis findings:")
        
        for attribute, metrics in self.results['fairness'].items():
            print(f"\nFindings for {attribute.title()}:")
            
            # Demographic parity
            dpd = metrics['demographic_parity_difference']
            if abs(dpd) < 0.05:
                parity_status = "good"
            elif abs(dpd) < 0.1:
                parity_status = "moderate"
            else:
                parity_status = "concerning"
            
            print(f"1. Demographic Parity: {parity_status.upper()} (Difference: {dpd:.4f})")
            print(f"   - This means the model's positive prediction rates " +
                 f"{'are similar' if parity_status == 'good' else 'differ'} across {attribute} groups.")
            
            # Equal opportunity
            eod = metrics['equalized_odds_difference']
            if abs(eod) < 0.05:
                eod_status = "good"
            elif abs(eod) < 0.1:
                eod_status = "moderate"
            else:
                eod_status = "concerning"
            
            print(f"2. Equalized Odds: {eod_status.upper()} (Difference: {eod:.4f})")
            print(f"   - This means the model's error rates " +
                 f"{'are similar' if eod_status == 'good' else 'differ'} across {attribute} groups.")
            
            # Performance metrics
            print(f"3. Performance disparities:")
            group_metrics = metrics['metrics_by_group']
            for metric in ['accuracy', 'precision', 'recall']:
                values = [group[metric] for group in group_metrics.values()]
                max_diff = max(values) - min(values)
                
                if max_diff < 0.05:
                    status = "MINIMAL"
                elif max_diff < 0.1:
                    status = "MODERATE"
                else:
                    status = "SUBSTANTIAL"
                
                print(f"   - {metric.title()} disparity: {status} (Max difference: {max_diff:.4f})")
        
        # Feature importance analysis
        print("\nFeature Importance Analysis:")
        
        for attribute, group_shap in self.results['shap'].items():
            print(f"\nFor {attribute.title()}:")
            
            # Calculate mean absolute SHAP values for each group
            group_top_features = {}
            for group, shap_values in group_shap.items():
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]  # Top 5 features
                
                # Get feature names
                top_feature_names = [self.encoded_feature_names[i] if i < len(self.encoded_feature_names) 
                                   else f"Feature_{i}" for i in top_features_idx]
                top_feature_values = mean_abs_shap[top_features_idx]
                
                group_top_features[group] = list(zip(top_feature_names, top_feature_values))
                
                print(f"  Top 5 important features for {group}:")
                for i, (feature, value) in enumerate(group_top_features[group]):
                    print(f"    {i+1}. {feature}: {value:.4f}")
            
            # Find differences in feature importance
            print(f"\n  Feature importance disparities for {attribute}:")
            
            all_top_features = set()
            for features in group_top_features.values():
                all_top_features.update([f for f, _ in features])
            
            # Create dictionary of feature importance by group
            feature_importance_by_group = {group: dict(features) for group, features in group_top_features.items()}
            
            # Find features with large disparities
            disparities = []
            for feature in all_top_features:
                values = []
                for group in group_top_features.keys():
                    group_dict = dict(group_top_features[group])
                    if feature in group_dict:
                        values.append(group_dict[feature])
                    else:
                        values.append(0)
                
                if len(values) > 1:
                    max_diff = max(values) - min(values)
                    if max_diff > 0.05:  # Only report significant disparities
                        disparities.append((feature, max_diff))
            
            # Report top disparities
            if disparities:
                disparities.sort(key=lambda x: x[1], reverse=True)
                print(f"  Top feature importance disparities:")
                for feature, diff in disparities[:3]:  # Show top 3 disparities
                    print(f"    - {feature}: {diff:.4f} difference between groups")
                    for group in group_top_features.keys():
                        group_dict = dict(group_top_features[group])
                        value = group_dict.get(feature, 0)
                        print(f"      {group}: {value:.4f}")
            else:
                print("  No significant feature importance disparities found.")
        
        # Overall recommendations
        print("\nRecommendations:")
        
        has_parity_issues = any(abs(metrics['demographic_parity_difference']) > 0.05 
                              for metrics in self.results['fairness'].values())
        has_odds_issues = any(abs(metrics['equalized_odds_difference']) > 0.05 
                            for metrics in self.results['fairness'].values())
        
        if has_parity_issues:
            print("1. Consider applying fairness constraints during model training:")
            print("   - Techniques like reweighing, prejudice remover, or fairness constraints can help")
            print("   - Explore libraries like Fairlearn or AIF360 for implementation")
        
        if has_odds_issues:
            print("2. Address disparate error rates:")
            print("   - Evaluate whether additional features could help reduce disparities")
            print("   - Consider post-processing techniques like equalized odds post-processing")
        
        print("3. Feature engineering and selection considerations:")
        print("   - Review features with high importance disparities across groups")
        print("   - Consider whether proxy variables might be introducing bias")
        
        print("4. Next steps for bias mitigation:")
        print("   - Regularly monitor fairness metrics during model development")
        print("   - Establish fairness thresholds for production models")
        print("   - Document fairness considerations and tradeoffs in model cards")

def main():
    """Main function to demonstrate FairLens Insight."""
    print("===== FairLens Insight: Model Fairness Analysis Tool =====\n")
    
    # Initialize FairLens
    fairlens = FairLensInsight(model_type="gradient_boosting")
    
    # Load and explore data
    fairlens.load_adult_census_data()
    
    # Train model
    fairlens.train_model()
    
    # Evaluate fairness
    fairlens.evaluate_fairness()
    
    # Analyze feature importance
    fairlens.analyze_feature_importance()
    
    # Generate fairness report
    fairlens.create_fairness_report()
    
    print("\n===== Analysis Complete =====")
    print("Visualization files have been saved to the current directory.")

if __name__ == "__main__":
    main()
