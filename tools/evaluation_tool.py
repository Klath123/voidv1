from crewai_tools import BaseTool
from typing import Dict, List
import re
from difflib import SequenceMatcher

class EvaluationTool(BaseTool):
    name: str = "Answer Evaluation Tool"
    description: str = "Evaluates student answers against reference answers"
    
    def _run(self, student_answers: List[Dict], 
             reference_answers: List[Dict]) -> dict:
        """
        Evaluate student answers
        student_answers: [{'q_num': 1, 'type': 'mcq', 'answer': 'A'}, ...]
        reference_answers: [{'q_num': 1, 'type': 'mcq', 'correct': 'A', 'marks': 1}, ...]
        """
        try:
            results = []
            total_marks = 0
            obtained_marks = 0
            
            for ref in reference_answers:
                q_num = ref['q_num']
                student_ans = next(
                    (a for a in student_answers if a['q_num'] == q_num), 
                    None
                )
                
                if not student_ans:
                    results.append({
                        'q_num': q_num,
                        'marks_obtained': 0,
                        'marks_total': ref['marks'],
                        'status': 'unanswered',
                        'confidence': 1.0
                    })
                    total_marks += ref['marks']
                    continue
                
                # Evaluate based on question type
                evaluation = self._evaluate_answer(
                    student_ans, ref
                )
                
                results.append(evaluation)
                total_marks += ref['marks']
                obtained_marks += evaluation['marks_obtained']
            
            return {
                'success': True,
                'total_marks': total_marks,
                'obtained_marks': obtained_marks,
                'percentage': (obtained_marks / total_marks * 100) if total_marks > 0 else 0,
                'question_results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _evaluate_answer(self, student_ans: Dict, ref: Dict) -> Dict:
        """Evaluate single answer based on type"""
        q_type = ref['type']
        
        if q_type == 'mcq':
            return self._evaluate_mcq(student_ans, ref)
        elif q_type == 'fill_blank':
            return self._evaluate_fill_blank(student_ans, ref)
        elif q_type == 'one_word':
            return self._evaluate_one_word(student_ans, ref)
        else:
            return {
                'q_num': ref['q_num'],
                'marks_obtained': 0,
                'marks_total': ref['marks'],
                'status': 'unsupported_type',
                'confidence': 0.0
            }
    
    def _evaluate_mcq(self, student_ans: Dict, ref: Dict) -> Dict:
        """Evaluate MCQ"""
        student = student_ans['answer'].strip().upper()
        correct = ref['correct'].strip().upper()
        
        is_correct = student == correct
        confidence = 1.0 if len(student) == 1 else 0.5
        
        return {
            'q_num': ref['q_num'],
            'marks_obtained': ref['marks'] if is_correct else 0,
            'marks_total': ref['marks'],
            'status': 'correct' if is_correct else 'incorrect',
            'confidence': confidence,
            'student_answer': student,
            'correct_answer': correct
        }
    
    def _evaluate_fill_blank(self, student_ans: Dict, ref: Dict) -> Dict:
        """Evaluate fill in the blank"""
        student = self._normalize_text(student_ans['answer'])
        correct_options = [
            self._normalize_text(ans) 
            for ans in ref.get('correct_options', [ref['correct']])
        ]
        
        # Check exact match
        if student in correct_options:
            return {
                'q_num': ref['q_num'],
                'marks_obtained': ref['marks'],
                'marks_total': ref['marks'],
                'status': 'correct',
                'confidence': 1.0,
                'student_answer': student_ans['answer']
            }
        
        # Check fuzzy match
        best_similarity = max(
            [SequenceMatcher(None, student, correct).ratio() 
             for correct in correct_options]
        )
        
        # Award marks based on similarity threshold
        if best_similarity >= 0.9:
            marks = ref['marks']
            status = 'correct'
        elif best_similarity >= 0.7:
            marks = ref['marks'] * 0.5
            status = 'partial'
        else:
            marks = 0
            status = 'incorrect'
        
        return {
            'q_num': ref['q_num'],
            'marks_obtained': marks,
            'marks_total': ref['marks'],
            'status': status,
            'confidence': best_similarity,
            'student_answer': student_ans['answer']
        }
    
    def _evaluate_one_word(self, student_ans: Dict, ref: Dict) -> Dict:
        """Evaluate one word answer"""
        return self._evaluate_fill_blank(student_ans, ref)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r'[^\w\s]', '', text.lower().strip())