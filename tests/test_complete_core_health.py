"""A completed notebook must not hide an absent/failed candidate channel."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_complete_core_health import check_events


class CoreHealthTests(unittest.TestCase):
    def events(self,pc=12,error=0):
        return [dict(data='\n'.join(['engine lists 12','engine lists 12',f'pubchem lists {pc}',f'errors {error} | popularity re-ranked lists 9',
            'PubChem channel','Engine 1 candidate lists','GLACIER + ICEBERG forward models','Two-engine fusion','Metric-identity fusion','Window-40 overlay','Execution receipt']))]
    def test_complete_actual_query_coverage_passes(self):self.assertTrue(check_events(self.events(),12)['complete_query_coverage'])
    def test_soft_pubchem_fallback_fails_even_with_other_stages_complete(self):
        events=self.events();events.append(dict(data='PUBCHEM CHANNEL FAILED -> base lists only: TimeoutError()'))
        with self.assertRaisesRegex(AssertionError,'fallback/failure'):check_events(events,12)
    def test_partial_pubchem_channel_fails(self):
        with self.assertRaisesRegex(AssertionError,'PubChem coverage'):check_events(self.events(pc=11),12)
    def test_missing_main_stage_fails(self):
        with self.assertRaisesRegex(AssertionError,'Main-engine errors'):check_events(self.events(error=1),12)
    def test_missing_simulator_completion_fails(self):
        events=self.events();events[0]['data']=events[0]['data'].replace('GLACIER + ICEBERG forward models','')
        with self.assertRaisesRegex(AssertionError,'Missing completed stage marker'):check_events(events,12)


if __name__=='__main__':unittest.main()
