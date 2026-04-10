export const mockApi = {
  parseResume: async (file) => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    return {
      name: "John Doe",
      email: "john.doe@email.com",
      phone: "+1-234-567-8900",
      skills: ["JavaScript", "React", "Node.js", "Python", "AWS", "Docker"],
      experience: [
        { title: "Senior Developer", company: "Tech Corp", duration: "2020-2023" },
        { title: "Developer", company: "StartUp Inc", duration: "2018-2020" }
      ],
      education: [{ degree: "BS Computer Science", school: "University", year: "2018" }]
    };
  },

  calculateATSScore: async (resume, jobDescription) => {
    await new Promise(resolve => setTimeout(resolve, 2000));
    return {
      score: 78,
      matchedKeywords: ["JavaScript", "React", "AWS", "Team collaboration"],
      missingKeywords: ["Kubernetes", "CI/CD", "Agile"],
      suggestions: [
        "Add more specific metrics to quantify achievements",
        "Include Kubernetes and CI/CD experience",
        "Highlight leadership and mentoring experience"
      ]
    };
  },

  chatWithResume: async (question, context) => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const responses = {
      default: "Based on your resume, you have 5 years of experience in software development with strong skills in JavaScript and React. You've worked at Tech Corp and StartUp Inc, progressing from Developer to Senior Developer."
    };
    return responses.default;
  },

  searchJobs: async (query, filters) => {
    try {
      const searchQuery = query || 'software developer';
      const response = await fetch(
        `https://serpapi.com/search.json?engine=google_jobs&q=${encodeURIComponent(searchQuery)}&api_key=demo`
      );
      const data = await response.json();
      
      if (data.jobs_results) {
        return data.jobs_results.slice(0, 10).map((job, index) => ({
          id: index + 1,
          title: job.title,
          company: job.company_name,
          location: job.location,
          salary: job.detected_extensions?.salary || 'Not specified',
          match: Math.floor(Math.random() * 20) + 75,
          posted: job.detected_extensions?.posted_at || 'Recently',
          description: job.description,
          link: job.share_link || job.related_links?.[0]?.link
        }));
      }
    } catch (error) {
      console.error('Job search error:', error);
    }
    
    return [
      {
        id: 1,
        title: "Senior React Developer",
        company: "Amazon",
        location: "Seattle, WA",
        salary: "$140k - $180k",
        match: 92,
        posted: "2 days ago"
      },
      {
        id: 2,
        title: "Full Stack Engineer",
        company: "Microsoft",
        location: "Redmond, WA",
        salary: "$130k - $170k",
        match: 85,
        posted: "1 week ago"
      },
      {
        id: 3,
        title: "Frontend Developer",
        company: "Google",
        location: "Mountain View, CA",
        salary: "$150k - $190k",
        match: 88,
        posted: "3 days ago"
      }
    ];
  },

  getApplications: async () => {
    await new Promise(resolve => setTimeout(resolve, 800));
    return [
      { id: 1, company: "Amazon", position: "Senior React Developer", status: "Interview", date: "2024-01-15" },
      { id: 2, company: "Microsoft", position: "Full Stack Engineer", status: "Applied", date: "2024-01-20" },
      { id: 3, company: "Google", position: "Frontend Developer", status: "Rejected", date: "2024-01-10" }
    ];
  },

  conductMockInterview: async (question) => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    const questions = [
      "Tell me about yourself and your experience.",
      "What's your greatest technical achievement?",
      "How do you handle tight deadlines?",
      "Describe a challenging bug you solved."
    ];
    return {
      question: questions[Math.floor(Math.random() * questions.length)],
      feedback: "Good answer! Consider adding more specific examples and metrics."
    };
  }
};
