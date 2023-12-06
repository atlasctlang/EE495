from torchvision.datasets import LFWPairs
import os


class LFWPairs(LFWPairs):

    def _get_pairs(self, images_dir: str):
        pair_names, data, targets = [], [], []
        with open(os.path.join(self.root, self.labels_file)) as f:
            lines = f.readlines()
            if self.split == "10fold":
                n_folds, n_pairs = lines[0].split("\t")
                n_folds, n_pairs = int(n_folds), int(n_pairs)
            else:
                n_folds, n_pairs = 1, int(lines[0])
            s = 1

            for fold in range(n_folds):
                matched_pairs = [line.strip().split("\t") for line in lines[s:s + n_pairs]]
                unmatched_pairs = [line.strip().split("\t") for line in lines[s + n_pairs:s + (2 * n_pairs)]]
                s += 2 * n_pairs
                for pair in matched_pairs:
                    img1, img2, same = self._get_path(pair[0], pair[1]), self._get_path(pair[0], pair[2]), 1
                    pair_names.append((pair[0], pair[0]))
                    data.append((img1, img2))
                    targets.append(same)
                for pair in unmatched_pairs:
                    img1, img2, same = self._get_path(pair[0], pair[1]), self._get_path(pair[2], pair[3]), -1
                    pair_names.append((pair[0], pair[2]))
                    data.append((img1, img2))
                    targets.append(same)

        return pair_names, data, targets
