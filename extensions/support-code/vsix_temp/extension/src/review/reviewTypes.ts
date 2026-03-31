export interface ReviewIssue {
    type: 'correctness' | 'security' | 'architecture' | 'performance';
    file: string;
    line: number;
    message: string;
}

export interface RequiredChange {
    file: string;
    instruction: string;
}

export interface ReviewerOutput {
    status: 'approve' | 'revise' | 'reject';
    issues: ReviewIssue[];
    required_changes: RequiredChange[];
}
