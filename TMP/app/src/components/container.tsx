interface ContainerProps {
  children: React.ReactNode;
  variant?: "default" | "topmargin" | "center";
}

const Container: React.FC<ContainerProps> = ({
  children,
  variant = "default",
}) => {
  const variants = {
    default: "h-dvh",
    topmargin: "pt-16 h-dvh",
    center: "h-dvh flex items-center justify-center",
  } as const;

  const variantClasses = variants[variant];

  return (
    <div className={`container mx-auto max-w-7xl ${variantClasses}`}>
      {children}
    </div>
  );
};

export default Container;
