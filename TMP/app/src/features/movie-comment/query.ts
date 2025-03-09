export const commentInclude = {
  user: {
    select: {
      id: true,
      name: true,
      image: true,
      username: true,
    },
  },
  mentions: {
    where: {
      deletedAt: null,
    },
    include: {
      user: {
        select: {
          id: true,
          name: true,
          username: true,
          image: true,
        },
      },
    },
  },
  _count: {
    select: {
      replies: true,
      favorites: {
        where: {
          deletedAt: null,
        },
      },
    },
  },
} as const;
