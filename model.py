import os
import math
import numpy as np
import pandas as pd


class PoissonProbabilityModel:

    def __init__(self, file_paths, adjust_home_advantage=False):
        """
        Class for calculating Poisson probability matrices.

        :param file_paths: Dictionary with names and paths to CSV files.
        :param adjust_home_advantage: Whether to account for home advantage.
        """
        self.file_paths = file_paths
        self.adjust_home_advantage = adjust_home_advantage
        self.data_frames = self._load_data()
        self.full_data = pd.concat(self.data_frames.values(), ignore_index=True)
        self.home_mean = self.full_data['home_score'].mean()
        self.away_mean = self.full_data['away_score'].mean()
        self.home_score_range = range(self.full_data['home_score'].min(), self.full_data['home_score'].max() + 1)
        self.away_score_range = range(self.full_data['away_score'].min(), self.full_data['away_score'].max() + 1)
        if self.adjust_home_advantage:
            self._adjust_home_mean()

    def _load_data(self):
        """
        Load data from files.
        """
        return {name: pd.read_csv(path).assign(pair=name) for name, path in self.file_paths.items()}

    def _adjust_home_mean(self):
        """
        Adjust the mean score for the home team to account for home advantage.
        """
        home_std = self.full_data['home_score'].std()
        away_std = self.full_data['away_score'].std()
        home_advantage = home_std - away_std
        self.home_mean += home_advantage

    @staticmethod
    def _poisson_probability(lmbda, k):
        """
        Calculate Poisson probability for k events given the mean lmbda.
        """
        return (np.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)

    def _generate_probability_matrix(self):
        """
        Generate a Poisson probability matrix.
        """
        prob_matrix = np.zeros((len(self.home_score_range), len(self.away_score_range)))

        for i, home_score in enumerate(self.home_score_range):
            for j, away_score in enumerate(self.away_score_range):
                home_prob = self._poisson_probability(self.home_mean, home_score)
                away_prob = self._poisson_probability(self.away_mean, away_score)
                prob_matrix[i, j] = home_prob * away_prob

        return prob_matrix

    def save_matrices_to_csv(self, output_dir):
        """
        Generate and save CSV files for each dataset.
        """
        for pair_name, df in self.data_frames.items():
            pair_mean_home = df['home_score'].mean()
            pair_mean_away = df['away_score'].mean()

            pair_matrix = np.zeros((len(self.home_score_range), len(self.away_score_range)))
            for i, home_score in enumerate(self.home_score_range):
                for j, away_score in enumerate(self.away_score_range):
                    home_prob = self._poisson_probability(pair_mean_home, home_score)
                    away_prob = self._poisson_probability(pair_mean_away, away_score)
                    pair_matrix[i, j] = home_prob * away_prob

            prob_df = pd.DataFrame(
                pair_matrix,
                index=[f"Home {i}" for i in self.home_score_range],
                columns=[f"Away {j}" for j in self.away_score_range],
            )
            output_path = f"{output_dir}/{pair_name}_scores.csv"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            prob_df.to_csv(output_path)
            print(f"Probability matrix for {pair_name} saved to {output_path}")

    def save_overall_matrix_to_csv(self, output_path):
        """
        Save the probability matrix for all combined data.
        """
        prob_matrix = self._generate_probability_matrix()
        prob_df = pd.DataFrame(
            prob_matrix,
            index=[f"Home {i}" for i in self.home_score_range],
            columns=[f"Away {j}" for j in self.away_score_range],
        )
        prob_df.to_csv(output_path)
        print(f"Overall probability matrix saved to {output_path}")


if __name__ == "__main__":
    ROOT_PATH = '/Users/olesyamba/PycharmProjects/betby_test_task_ds'
    DATA_FOLDER = os.path.join(ROOT_PATH, 'data')

    files = ("test_data_first_pair.csv", "test_data_second_pair.csv", "test_data_third_pair.csv")

    file_paths = {
        f"{i}_pair": os.path.join(DATA_FOLDER, file) for i, file in enumerate(files)
    }

    calculator = PoissonProbabilityModel(file_paths, adjust_home_advantage=True)
    calculator.save_matrices_to_csv(output_dir="scores")
    calculator.save_overall_matrix_to_csv(output_path="scores/overall_scores.csv")