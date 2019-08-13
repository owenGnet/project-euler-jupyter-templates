import itertools as it
import json
import logging
import pickle
from collections import deque, namedtuple
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

import src.euler_to_jupyter as sut

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'test_data'

print('DATA_DIR', DATA_DIR)

JUPYTER_TEMPLATE = """\
{
 "cells": [
  {
   "source": []
}
],
 "metadata": {
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
"""

RECENT_PROBLEMS_PAGE = '''
<table id="problems_table" class="grid">
<tbody><tr><th class="id_column">ID</th><th class="title_column">Description / Title</th><th class="solved_by_column">Solved By</th></tr>
<tr><td class="id_column">5</td><td><a href="problem=5" title="Published on Saturday, 29th June 2019, 07:00 pm">Coloured Graphs</a></td><td><div style="text-align:center;">58</div></td></tr>
<tr><td class="id_column">4</td><td><a href="problem=4" title="Published on Sunday, 2nd June 2019, 07:00 am">Beds and Desks</a></td><td><div style="text-align:center;">165</div></td></tr>
<tr><td class="id_column">3</td><td><a href="problem=3" title="Published on Sunday, 26th May 2019, 04:00 am">One more one</a></td><td><div style="text-align:center;">147</div></td></tr>
<tr><td class="id_column">2</td><td><a href="problem=2" title="Published on Sunday, 19th May 2019, 01:00 am">Colouring a Loop</a></td><td><div style="text-align:center;">92</div></td></tr>
<tr><td class="id_column">1</td><td><a href="problem=1" title="Published on Sunday, 19th May 2019, 01:00 am">Colouring a Strip</a></td><td><div style="text-align:center;">188</div></td></tr>
</tbody></table>
'''


def test_prep_local_env_exceptions():
    with pytest.raises(AttributeError):
        sut.prep_local_env('this-is-not-a-Path-object')
    with pytest.raises(FileNotFoundError):
        sut.prep_local_env(Path('this-is-not-an-existing-Path'))
    with pytest.raises(FileNotFoundError):
        sut.prep_local_env(Path(DATA_DIR))


@mock.patch('pathlib.Path.mkdir')
@mock.patch('pathlib.Path.exists')
@mock.patch('pathlib.Path.is_file')
def test_prep_local_env_mkdir(mocked_path_is_file, mocked_path_exists, mocked_path_mkdir):
    mocked_path_is_file.return_value = True
    mocked_path_exists.return_value = False
    sut.prep_local_env(Path('_'))
    assert mocked_path_mkdir.call_count == 1


@mock.patch('builtins.open')
def test_get_template_data(mocked_open):
    pth = Path('_')
    mocked_open.side_effect = [StringIO(JUPYTER_TEMPLATE),
                               StringIO('not-valid-json'),
                               StringIO(JUPYTER_TEMPLATE.replace('cells', '_')),
                               StringIO(JUPYTER_TEMPLATE.replace('metadata', '_')),
                               ]
    assert sut.get_template_data(pth) == json.loads(JUPYTER_TEMPLATE)
    with pytest.raises(json.decoder.JSONDecodeError):
        sut.get_template_data(pth)
    with pytest.raises(KeyError):
        sut.get_template_data(pth)
    with pytest.raises(KeyError):
        sut.get_template_data(pth)


@mock.patch('src.euler_to_jupyter.download_images')
@mock.patch('src.euler_to_jupyter.get_the_response')
def test_populate_problems(mocked_get_response, mocked_download_images):
    p20_response = pickle.load(open(DATA_DIR / 'p20_response.pickle', 'rb'))
    p333_response = pickle.load(open(DATA_DIR / 'p333_response.pickle', 'rb'))
    mocked_get_response.side_effect = [p20_response, p333_response]
    mocked_download_images.return_value = True

    pnums = (20, 333)
    problems_out = sut.populate_problems(','.join([str(n) for n in pnums]))
    probs_out_baseline = pickle.load(open(DATA_DIR / 'p20_333_populated_problems.pickle', 'rb'))
    assert tuple(problems_out.keys()) == pnums
    assert problems_out.keys() == probs_out_baseline.keys()
    assert [v.keys() for v in problems_out.values()] == [{'content_raw': _}.keys() for _ in range(2)]
    assert [(v['content_raw'].name) for v in problems_out.values()] == ['div', 'div']
    assert problems_out == probs_out_baseline


@mock.patch('src.euler_to_jupyter.download_images')
@mock.patch('src.euler_to_jupyter.get_the_response')
def test_populate_problems_throw_value_error(mocked_get_response, mocked_download_images):
    p20_response = pickle.load(open(DATA_DIR / 'p20_response.pickle', 'rb'))
    p333_response = pickle.load(open(DATA_DIR / 'p333_response.pickle', 'rb'))
    mocked_get_response.side_effect = [p20_response, p333_response]
    mocked_download_images.return_value = True

    with pytest.raises(ValueError):
        _ = sut.populate_problems('blurg')


def test_process_problems():
    problems_in = pickle.load(open(DATA_DIR / 'p20_333_populated_problems.pickle', 'rb'))
    processed = sut.process_problems(problems_in)

    pnums = (20, 333)
    assert tuple(processed.keys()) == pnums
    #    processed.keys() == {pnum: _ for pnum in pnums}.keys()
    assert [v.keys() for v in processed.values()] == [{'content_raw': _, 'content_final': _}.keys() for _ in range(2)]
    assert all(v['content_final'].startswith('<h2') for v in processed.values())
    assert all(v['content_final'].endswith('</div>') for v in processed.values())

    processed_baseline = pickle.load(open(DATA_DIR / 'p20_333_processed_problems.pickle', 'rb'))
    assert processed.keys() == processed_baseline.keys()
    assert processed == processed_baseline


@mock.patch('src.euler_to_jupyter.process_problems')
@mock.patch('src.euler_to_jupyter.populate_problems')
def test_populate_cell_pair_q(mocked_populate_problems, mocked_process_problems):
    populated_problems = pickle.load(open(DATA_DIR / 'p20_333_populated_problems.pickle', 'rb'))
    mocked_populate_problems.return_value = populated_problems
    processed_problems = pickle.load(open(DATA_DIR / 'p20_333_processed_problems.pickle', 'rb'))
    mocked_process_problems.return_value = processed_problems

    q = sut.populate_cell_pair_q('_')
    assert len(q) == 2
    # cell_pair_baseline = pickle.dump(q, open(DATA_DIR / 'p20_333_cell_pair_q.pickle', 'wb'))
    cell_pair_baseline = pickle.load(open(DATA_DIR / 'p20_333_cell_pair_q.pickle', 'rb'))

    first_baseline = cell_pair_baseline.popleft()
    assert q.count(first_baseline) == 1
    assert q.index(first_baseline) == 0
    first_test = q.popleft()
    assert first_baseline == first_test
    last_baseline = cell_pair_baseline.popleft()
    last_test = q.popleft()
    assert last_baseline == last_test
    assert len(q) == 0

def test_gather_notebooks_default():
    DEFAULT_PROBEMS_PER_NOTEBOOK = 10
    cell_pair_20_333 = pickle.load(open(DATA_DIR / 'p20_333_cell_pair_q.pickle', 'rb'))
    json_template = json.loads(JUPYTER_TEMPLATE)
    cell_pairs = it.cycle(cell_pair_20_333)

    num_cell_pairs = 2  # 4 actual cells
    some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
    gen = sut.gather_notebooks(json_template, some_cell_pairs)
    notebooks = [ntbk for ntbk in gen]
    assert len(notebooks) == 1
    # template cell + (2 * (1 problem, 1 answer cell in each cell pair))
    assert len(notebooks[0]['cells']) == 1 + (num_cell_pairs * 2)
    assert notebooks[0].keys() == json_template.keys()

    num_cell_pairs = 4  # 8 actual cells in one notebook + template cell
    some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
    json_template = json.loads(JUPYTER_TEMPLATE)
    gen = sut.gather_notebooks(json_template, some_cell_pairs)
    notebooks = [ntbk for ntbk in gen]
    assert len(notebooks) == 1
    assert len(notebooks[0]['cells']) == 1 + (num_cell_pairs * 2)

    num_cell_pairs = 10  # 20 non-template cells, should fit in one notebook
    some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
    json_template = json.loads(JUPYTER_TEMPLATE)
    gen = sut.gather_notebooks(json_template, some_cell_pairs)
    notebooks = [ntbk for ntbk in gen]
    assert len(notebooks) == 1
    assert len(notebooks[0]['cells']) == 1 + (min(num_cell_pairs, DEFAULT_PROBEMS_PER_NOTEBOOK) * 2)

    num_cell_pairs = 11  # 20 non-template cells in one notebook, 2 in a second notebook
    some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
    json_template = json.loads(JUPYTER_TEMPLATE)
    gen = sut.gather_notebooks(json_template, some_cell_pairs)
    notebooks = [ntbk for ntbk in gen]
    assert len(notebooks) == 2
    assert len(notebooks[0]['cells']) == 1 + (min(num_cell_pairs, DEFAULT_PROBEMS_PER_NOTEBOOK) * 2)
    assert len(notebooks[1]['cells']) == 3

    num_cell_pairs_test_nums = [11, 19, 22, 28, 31]
    for num_cell_pairs in num_cell_pairs_test_nums:
        some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
        json_template = json.loads(JUPYTER_TEMPLATE)
        gen = sut.gather_notebooks(json_template, some_cell_pairs)
        notebooks = [ntbk for ntbk in gen]
        a, b = divmod(num_cell_pairs, DEFAULT_PROBEMS_PER_NOTEBOOK)
        assert len(notebooks) == a + 1
        for i in range(a):
            assert len(notebooks[i]['cells']) == 1 + (min(num_cell_pairs, DEFAULT_PROBEMS_PER_NOTEBOOK) * 2)
        assert len(notebooks[-1]['cells']) == 1 + (b * 2)


def test_gather_notebooks_n_problems():
    cell_pair_20_333 = pickle.load(open(DATA_DIR / 'p20_333_cell_pair_q.pickle', 'rb'))
    json_template = json.loads(JUPYTER_TEMPLATE)
    cell_pairs = it.cycle(cell_pair_20_333)

    problems_per_notebook = 4
    num_cell_pairs = 5  # 10 actual cells
    some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
    logging.info(f'TESTING: {problems_per_notebook} problems per notebook, {num_cell_pairs} total problems')
    gen = sut.gather_notebooks(json_template, some_cell_pairs, problems_per_notebook=problems_per_notebook)
    notebooks = [ntbk for ntbk in gen]
    assert len(notebooks) == 2
    assert len(notebooks[0]['cells']) == 1 + (min(num_cell_pairs, problems_per_notebook) * 2)
    assert len(notebooks[1]['cells']) == 3

    problems_per_notebook = 3
    num_cell_pairs = 11  # 10 actual cells
    some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
    logging.info(f'TESTING: {problems_per_notebook} problems per notebook, {num_cell_pairs} total problems')
    gen = sut.gather_notebooks(json_template, some_cell_pairs, problems_per_notebook=problems_per_notebook)
    notebooks = [ntbk for ntbk in gen]
    assert len(notebooks) == 4
    assert len(notebooks[0]['cells']) == 1 + (min(num_cell_pairs, problems_per_notebook) * 2)
    assert len(notebooks[1]['cells']) == 7

    problems_per_notebook_test_nums = [3, 8, 15, 20, 22]
    num_cell_pairs_test_nums = [13, 19, 22, 28, 31]

    test_args = it.product(problems_per_notebook_test_nums, num_cell_pairs_test_nums)
    for pnum_limit, cp_nums in test_args:
        logging.info(f'TESTING: {pnum_limit} problems per notebook, {cp_nums} total problems')
        some_cell_pairs = deque(it.islice(cell_pairs, 0, num_cell_pairs))
        json_template = json.loads(JUPYTER_TEMPLATE)
        gen = sut.gather_notebooks(json_template, some_cell_pairs, problems_per_notebook=pnum_limit)
        notebooks = [ntbk for ntbk in gen]
        a, b = divmod(num_cell_pairs, pnum_limit)
        assert len(notebooks) == a + 1
        for i in range(a):
            assert len(notebooks[i]['cells']) == 1 + (min(num_cell_pairs, pnum_limit) * 2)
        assert len(notebooks[-1]['cells']) == 1 + (b * 2)


@mock.patch('requests.get')
def test_parse_input_rand_5(mocked_get):
    Response = namedtuple('Response', 'text')
    resp = Response(RECENT_PROBLEMS_PAGE)
    recent_page_url = 'https://projecteuler.net/recent'

    mocked_get.side_effect = lambda x: {
        recent_page_url: resp,
    }.get(x, 'unhandled request {}'.format(x))

    pnums = sut.parse_input('rand,5')
    # given numbers 1 thru 5 in mocked html, requesting 5 random problems should return all of them
    assert sorted(pnums) == list(range(1, 6))


@mock.patch('requests.get')
def test_parse_input_rand_failed_request(mocked_get):
    Response = namedtuple('Response', 'text')
    bad_resp = Response('blurg>')
    recent_page_url = 'https://projecteuler.net/recent'

    default_max_pnum = 677
    mocked_get.side_effect = lambda x: {
           recent_page_url: bad_resp
    }.get(x, 'unhandled request {}'.format(x))

    pnums = sut.parse_input('rand,{}'.format(default_max_pnum))
    # BeautifulSoup fails (NoneType) -> defaults to using 677 as most-recent-problem, all 677 ints should be returned
    assert sorted(pnums) == list(range(1, (default_max_pnum + 1)))


def test_parse_input_csv():
    pnums = sut.parse_input('5,7,1')
    assert pnums == [1, 5, 7]
    pnums = sut.parse_input('99')
    assert pnums == [99]
    pnums = sut.parse_input('5,6,123')
    assert pnums == [5, 6, 123]


def test_parse_input_range():
    pnums = sut.parse_input('1-100')
    assert pnums == list(range(1, 101))
    pnums = sut.parse_input('5-7,11-55')
    assert pnums == list(range(5, 8)) + list(range(11, 56))
    pnums = sut.parse_input('5-7,11-55,1-3')
    assert pnums == [1, 2, 3] + list(range(5, 8)) + list(range(11, 56))


def test_parse_input_mixed():
    pnums = sut.parse_input('1-100,5,200')
    assert pnums == list(range(1, 101)) + [200, ]
    pnums = sut.parse_input('5,200,300-305,444')
    assert pnums == [5, 200, 300, 301, 302, 303, 304, 305, 444]
    pnums = sut.parse_input('333,2-4,222,5-7')
    assert pnums == list(range(2, 8)) + [222, 333]
