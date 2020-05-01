import pandas as pd


class RefitPolicy:
    _sort_func_name = "final_score_sort"

    def __init__(self):
        self.filter_funcs = []
        self.assignments = []
        self.sorting_func = None

    def filter(self, func):
        self.filter_funcs.append(func)
        return self

    def assign(self, **kwargs):
        if self._sort_func_name in kwargs.keys():
            raise ValueError(f"We use {self._sort_func_name} internally, cannot use this column name")
        self.assignments.append(kwargs)
        return self

    def sort(self, func):
        self.sorting_func = func
        return self

    def pick_best_estimator(self, gridsearch_obj):
        dataf = pd.DataFrame(gridsearch_obj.cv_results_)
        for filter in self.filter_funcs:
            dataf = dataf.loc[filter]
        for assignment in self.assignments:
            for name, func in assignment.items():
                dataf = dataf.assign(*{name: func})
        dataf = dataf.assign(*{self._sort_func_name: self.sorting_func})
        return dataf.sort_values(self._sort_func_name, ascending=False)
