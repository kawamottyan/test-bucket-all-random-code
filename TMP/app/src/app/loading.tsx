import Image from "next/image";

const Loading = () => {
  return (
    <div className="flex h-dvh w-full flex-col items-center justify-center gap-4">
      <div className="relative h-12 w-12">
        <Image src="/beaker_white.svg" alt="Loading animation" fill priority />
      </div>
    </div>
  );
};
export default Loading;
