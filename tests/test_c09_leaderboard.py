import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from c09_leaderboard import summarize


class SilverRankTests(unittest.TestCase):
    def test_matching_boundary_score_does_not_imply_qualifying_rank(self):
        rows = [dict(Rank=str(i), Score='.482', TeamMemberUserNames='other') for i in range(1, 1001)]
        rows[79]['TeamMemberUserNames'] = 'clarkkitchen'
        result = summarize(rows)
        self.assertEqual(result['nominal_silver_cutoff_rank'], 50)
        self.assertFalse(result['provisional_silver_qualifying_position'])
        self.assertFalse(result['final_competition_medal'])
        rows[79]['TeamMemberUserNames'] = 'other'; rows[39]['TeamMemberUserNames'] = 'clarkkitchen'
        self.assertTrue(summarize(rows)['provisional_silver_qualifying_position'])


if __name__ == '__main__':
    unittest.main()
