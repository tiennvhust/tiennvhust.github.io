/**
 * ============================================================================
 *  YOUR CV LIVES HERE.
 * ============================================================================
 *  To update the website, edit the data below — you almost never need to touch
 *  the HTML/components again. Each section is a typed list; add, remove, or
 *  reorder entries and the page rebuilds itself.
 *
 *  Tip: anything typed as `string` can contain inline HTML (e.g. links) if you
 *  need it — see the `html` helper usage in the project descriptions.
 * ============================================================================
 */

export interface Profile {
  name: string;
  tagline: string;
  /** Path (under /public) to the downloadable CV/résumé PDF. */
  resumePdf: string;
  /** URL of the embedded chatbot (Streamlit app). Leave empty ("") to hide it. */
  chatbotUrl: string;
}

export interface ContactLink {
  /** Full Font Awesome icon class, e.g. "fa-solid fa-envelope" or "fa-brands fa-linkedin". */
  icon: string;
  /** Optional brand color for the icon. */
  color: string;
  label: string;
  /** href — use "mailto:" for email, or a full URL. Empty for plain text. */
  href: string;
}

export interface EducationEntry {
  institution: string;
  location: string;
  period: string;
  degree: string;
  note?: string;
}

export interface ExperienceEntry {
  company: string;
  location: string;
  period: string;
  role: string;
  summary: string;
  /** Bullet points. Each may contain inline HTML (links etc.). */
  details: string[];
}

export interface ProjectMedia {
  type: 'image' | 'video';
  src: string;
  caption?: string;
  /** For videos hosted remotely you may pass an absolute URL in `src`. */
}

export interface ProjectBlock {
  heading: string;
  /** Paragraphs (inline HTML allowed). */
  paragraphs?: string[];
  media?: ProjectMedia[];
}

export interface ProjectEntry {
  year: string;
  /** Small label above the title, e.g. "Master's Thesis". */
  kicker?: string;
  title: string;
  blocks: ProjectBlock[];
}

export interface Publication {
  /** Full citation; wrap your own name in <strong> to bold it. */
  citation: string;
  doiLabel?: string;
  doiUrl?: string;
}

export const profile: Profile = {
  name: 'Tien Nguyen',
  tagline:
    'Systems software engineer with hands-on experience across the low-level stack with a strong grounding in operating-system internals. Comfortable in C, C++, Go, Rust, Python, and Matlab.',
  resumePdf: '/files/tienn.pdf',
  chatbotUrl: 'https://tienschatbot.streamlit.app/?embed=true',
};

export const contacts: ContactLink[] = [
  {
    icon: 'fa-solid fa-envelope',
    color: '#e0e0e0',
    label: 'tiennguyen.ie@outlook.com',
    href: 'mailto:tiennguyen.ie@outlook.com',
  },
  {
    icon: 'fa-brands fa-linkedin',
    color: '#0A66C2',
    label: 'LinkedIn',
    href: 'https://www.linkedin.com/in/tienvn',
  },
  {
    icon: 'fa-brands fa-github',
    color: '#e0e0e0',
    label: 'GitHub',
    href: 'https://github.com/tiennvhust',
  },
];

export const education: EducationEntry[] = [
  {
    institution: 'University College Cork',
    location: 'Cork, Ireland',
    period: 'May 2022 – May 2024',
    degree: 'MSc — Electrical and Electronics Engineering',
    note: 'Graduated with First Class Honours',
  },
  {
    institution: 'Technical University of Munich',
    location: 'Munich, Germany',
    period: 'Oct 2019 – Apr 2020',
    degree: 'Exchange Student',
  },
  {
    institution: 'Hanoi University of Science and Technology',
    location: 'Hanoi, Vietnam',
    period: 'Oct 2015 – Aug 2020',
    degree: 'Engineer — Control Engineering and Automation',
    note: 'Graduated with GPA 3.37 / 4.0',
  },
];

export const experience: ExperienceEntry[] = [
  {
    company: 'Qualcomm Technologies',
    location: 'Cork, Ireland',
    period: 'May 2024 – Present',
    role: 'Systems Software Engineer',
    summary:
      'Develop machine learning and signal processing solutions for mobile and wearables platforms.',
    details: [
    ],
  },
  {
    company: 'Vietnam-Korea Institute of Science and Technology',
    location: 'Hanoi, Vietnam',
    period: 'Aug 2021 – Apr 2022',
    role: 'Robotics Software Engineer',
    summary:
      'Developed software for an omnidirectional robot prototype.',
    details: [
    ],
  },
  {
    company: 'Viettel High Technology Industries Corporation',
    location: 'Hanoi, Vietnam',
    period: 'Nov 2020 – Aug 2021',
    role: 'Embedded Software Engineer',
    summary:
      'Developed and maintained Linux kernel device drivers for custom telecom hardware.',
    details: [
    ],
  },
];

export const projects: ProjectEntry[] = [
  {
    year: '2026',
    kicker: 'Personal Project',
    title: 'eBPF-Based GPU Observability',
    blocks: [
      {
        heading: 'Description',
        paragraphs: [
          'A zero-instrumentation CUDA observability tool with eBPF, reporting device-memory leaks, kernel-launch statistics, and host/device transfer volume.',
        ],
      },
    ],
  },
  {
    year: '2025',
    kicker: 'Personal Project',
    title: 'RAG-Based Profile Assistant',
    blocks: [
      {
        heading: 'Description',
        paragraphs: [
          'Built a domain-specific chatbot using Llama 3 and Sentence-Transformers to answer recruiter queries with high precision.',
        ],
      },
      {
        heading: 'Contributions',
        paragraphs: [
          'Hybrid Retrieval Architecture: engineered a custom retrieval engine combining semantic vector search (cosine similarity) with keyword filtering to retrieve relevant context from a flattened "atomic fact" JSON dataset.',
          'Hallucination Prevention: designed a deterministic "fact injection" pipeline that pre-calculates quantitative metrics (e.g. years of experience) in Python, ensuring 100% numerical accuracy in LLM responses.',
          'Semantic Guardrails: implemented a local intent-classification router (BERT-based) to filter out-of-scope queries client-side, reducing external API costs and latency.',
        ],
      },
    ],
  },
  {
    year: '2023',
    kicker: "Master's Thesis",
    title:
      'Low-Power & Real-Time Seizure Monitoring Using AI-Assisted Sonification of Neonatal EEG',
    blocks: [
      {
        heading: 'Description',
        paragraphs: [
          'This study aims to develop a practical application to assist in EEG diagnostic procedures based on previous work. We target a battery-powered and inexpensive system that can be widely accessible in underdeveloped regions where neonates are the most vulnerable due to the scarcity of expertise and equipment.',
          'A real-time processing design was presented as part of this work, allowing the sonification algorithm to operate and synthesize audio in parallel with EEG acquisition. The new design contains essential modifications without affecting its consistency with the original study. This approach removes the need for human intervention in the process and thus provides better accessibility for medical workers.',
          'Bringing artificial intelligence to the far edge is another step toward realizing the targeted application. The effects of full integer quantization on neural networks were studied. Although the conversion into integer representations results in a slight accuracy loss, it significantly improves the inference speed, memory footprint, power consumption, and hardware adaptability of deep learning models.',
          'An implementation of the proposed system was presented using a low-power and AI-capable microcontroller. Its speed and energy consumption are linearly correlated with the number of input EEG channels. This real-time EEG analysis-assisting system can process up to 55 parallel channels while being powered by a small battery.',
        ],
      },
      {
        heading: 'Algorithm Design',
        media: [
          {
            type: 'image',
            src: '/images/design.png',
            caption: 'Figure 1: Real-Time Algorithm Design',
          },
        ],
        paragraphs: [
          'Parallelization is a crucial technique to reduce overall computation time and enable more efficient processing. Figure 1 presents a multi-threading design for the AI-assisted sonification algorithm where its tasks are set up for concurrent and real-time operation.',
        ],
      },
      {
        heading: 'Implementation Results',
        media: [
          {
            type: 'image',
            src: '/images/power.png',
            caption: 'Figure 2: Power Consumption by Number of Processing Channels',
          },
        ],
        paragraphs: [
          'Figure 2 shows the measurement results in error bars with lower and upper limits corresponding to best- and worst-case scenarios, respectively. Overall, the power consumption of the algorithm has a linear correlation with the number of operating EEG channels.',
        ],
      },
    ],
  },
  {
    year: '2021',
    kicker: 'VKIST — Research & Development',
    title: 'Omnidirectional Robot Prototype Development',
    blocks: [
      {
        heading: 'Description',
        paragraphs: [
          'Developing an omnidirectional robot prototype for research and development of a navigation system for field robots.',
        ],
      },
      {
        heading: 'Contributions',
        paragraphs: [
          'I designed the electrical system, safety management system, actuator control software, and graphical user interface.',
        ],
      },
      {
        heading: 'Actuator Control Software',
        paragraphs: [
          'I designed and developed software to control BLDC motor drivers, which drive the robot wheels. Control and query commands are sent to the motor driver via a UART communication port. The main language used was C/C++ with the Robot Operating System (ROS). Source code: <a href="https://github.com/tiennvhust/bldc_driver">here</a>.',
        ],
        media: [
          { type: 'video', src: '/videos/1683561766185.MP4' },
        ],
      },
      {
        heading: 'Graphical User Interface',
        paragraphs: [
          'I designed and developed the robot graphical user interface (GUI) using Qt Creator. This GUI lets users control the robot via a software joystick and displays the robot\'s data and status. The main language used was C/C++ with Qt Creator and ROS. Source code: <a href="https://github.com/tiennvhust/vk_omni_gui">here</a>.',
        ],
        media: [
          {
            type: 'video',
            src: 'https://user-images.githubusercontent.com/95061513/160900081-84f24e7f-518d-42c9-aed3-01580d267aa2.mp4',
          },
        ],
      },
      {
        heading: 'Safety Management System',
        paragraphs: [
          'I designed a safety mechanism using a PLC and software to control the operation of the robot. The development of this system involved PLC programming and C/C++ with ROS. Source code: <a href="https://github.com/tiennvhust/vk_omni_plc">here</a>.',
        ],
      },
    ],
  },
];

export const publications: Publication[] = [
  {
    citation:
      '<strong>T. Nguyen</strong>, A. Daly, S. Gomez-Quintana, F. O\'Sullivan, A. Temko and E. Popovici, "Low-Power Real-Time Seizure Monitoring Using AI-Assisted Sonification of Neonatal EEG," in IEEE Transactions on Emerging Topics in Computing, vol. 13, no. 1, pp. 80–89, Jan.–March 2025',
    doiLabel: '10.1109/TETC.2024.3481035',
    doiUrl: 'https://doi.org/10.1109/TETC.2024.3481035',
  },
  {
    citation:
      '<strong>T. Van Nguyen</strong>, A. Daly, F. O\'Sullivan, S. G. Quintana, A. Temko and E. Popovici, "A real-time and ultra-low power implementation of an AI-assisted sonification algorithm for neonatal EEG," 2023 9th International Workshop on Advances in Sensors and Interfaces (IWASI), Monopoli (Bari), Italy, 2023, pp. 313–318',
    doiLabel: '10.1109/IWASI58316.2023.10164463',
    doiUrl: 'https://doi.org/10.1109/IWASI58316.2023.10164463',
  },
];
